#include "fsr2_backend_gl.hpp"
#include <ffx_fsr2.h>
#include <ffx_fsr2_private.h>

#include <cstring>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

#include <gl/shader_include.hpp>
#include <gllib/log.hpp>

#define GFX_LOG_ERROR(...) gllib::logf(gllib::LogLevel::error, __VA_ARGS__)
#define GFX_LOG_WARN(...)  gllib::logf(gllib::LogLevel::warn, __VA_ARGS__)

namespace {

constexpr uint32_t FSR2_MAX_QUEUED_FRAMES          = 4;
constexpr uint32_t FSR2_MAX_RESOURCE_COUNT         = 64;
constexpr uint32_t FSR2_MAX_STAGING_RESOURCE_COUNT = 8;
constexpr uint32_t FSR2_MAX_GPU_JOBS               = 32;
constexpr uint32_t FSR2_MAX_UNIFORM_BUFFERS        = 4;
constexpr uint32_t FSR2_MAX_IMAGE_VIEWS            = 32;
constexpr uint32_t FSR2_MAX_BUFFERED_DESCRIPTORS  = FFX_FSR2_PASS_COUNT * FSR2_MAX_QUEUED_FRAMES;
constexpr uint32_t FSR2_UBO_RING_BUFFER_SIZE       = FSR2_MAX_BUFFERED_DESCRIPTORS * FSR2_MAX_UNIFORM_BUFFERS;
constexpr uint32_t FSR2_UBO_SIZE                   = 256;
constexpr uint32_t FSR2_DEFAULT_SUBGROUP_SIZE      = 32;

struct Texture { GLuint id = {}; };
struct Buffer  { GLuint id = {}; };
struct Sampler { GLuint id = {}; };

struct BackendContext_GL {
    enum class Aspect { UNDEFINED, COLOR, DEPTH };

    struct Resource {
        FfxResourceDescription resourceDescription;
        Buffer buffer = {};
        Texture textureAllMipsView = {};
        Texture textureSingleMipViews[FSR2_MAX_IMAGE_VIEWS] = {};
        Aspect textureAspect = {};
    };

    struct UniformBuffer {
        Buffer bufferResource = {};
        uint8_t* pData = {};
    };

    FfxDeviceCapabilities capabilities = {};

    uint32_t gpuJobCount = 0;
    FfxGpuJobDescription gpuJobs[FSR2_MAX_GPU_JOBS] = {};

    uint32_t nextStaticResource = 0;
    uint32_t nextDynamicResource = 0;
    uint32_t stagingResourceCount = 0;
    Resource resources[FSR2_MAX_RESOURCE_COUNT] = {};
    FfxResourceInternal stagingResources[FSR2_MAX_STAGING_RESOURCE_COUNT] = {};

    Sampler pointSampler = {};
    Sampler linearSampler = {};

    UniformBuffer uboRingBuffer[FSR2_UBO_RING_BUFFER_SIZE] = {};
    uint32_t uboRingBufferIndex = 0;

    // Shader include directories for GLSL compilation
    std::vector<std::string> shaderIncludeDirs;
};

// ─── pass → .glsl filename mapping ────────────────────────────────────────
static const char* passGLSLFile(FfxFsr2Pass pass) {
    switch (pass) {
        case FFX_FSR2_PASS_DEPTH_CLIP:                    return "ffx_fsr2_depth_clip_pass.glsl2";
        case FFX_FSR2_PASS_RECONSTRUCT_PREVIOUS_DEPTH:    return "ffx_fsr2_reconstruct_previous_depth_pass.glsl2";
        case FFX_FSR2_PASS_LOCK:                           return "ffx_fsr2_lock_pass.glsl2";
        case FFX_FSR2_PASS_ACCUMULATE:
        case FFX_FSR2_PASS_ACCUMULATE_SHARPEN:            return "ffx_fsr2_accumulate_pass.glsl2";
        case FFX_FSR2_PASS_RCAS:                           return "ffx_fsr2_rcas_pass.glsl2";
        case FFX_FSR2_PASS_COMPUTE_LUMINANCE_PYRAMID:     return "ffx_fsr2_compute_luminance_pyramid_pass.glsl2";
        case FFX_FSR2_PASS_GENERATE_REACTIVE:              return "ffx_fsr2_autogen_reactive_pass.glsl2";
        case FFX_FSR2_PASS_TCR_AUTOGENERATE:               return "ffx_fsr2_tcr_autogen_pass.glsl2";
        default: return nullptr;
    }
}

// ─── permutation defines — also builds macro table for include resolver ──
struct Fsr2PermutationDefines {
    std::string glslText;
    std::map<std::string,int> macros;
};

static Fsr2PermutationDefines buildPermutationDefines(const FfxPipelineDescription* desc, FfxFsr2Pass pass, const FfxDeviceCapabilities& caps) {
    Fsr2PermutationDefines result;
    auto add = [&](const char* name, int val) {
        result.glslText += "#define "; result.glslText += name; result.glslText += ' '; result.glslText += std::to_string(val); result.glslText += '\n';
        result.macros[name] = val;
    };

    add("FFX_GPU", 1);
    add("FFX_GLSL", 1);

    bool hdr        = (desc->contextFlags & FFX_FSR2_ENABLE_HIGH_DYNAMIC_RANGE) != 0;
    bool lowResMV   = (desc->contextFlags & FFX_FSR2_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) == 0;
    bool jitterMV   = (desc->contextFlags & FFX_FSR2_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION) != 0;
    bool depthInv   = (desc->contextFlags & FFX_FSR2_ENABLE_DEPTH_INVERTED) != 0;
    bool sharpen    = (pass == FFX_FSR2_PASS_ACCUMULATE_SHARPEN);

    bool fp16 = caps.fp16Supported && pass != FFX_FSR2_PASS_RCAS;
    bool useLut = caps.waveLaneCountMax == 64;

    add("FFX_FSR2_OPTION_HDR_COLOR_INPUT",              hdr      ? 1 : 0);
    add("FFX_FSR2_OPTION_LOW_RESOLUTION_MOTION_VECTORS", lowResMV ? 1 : 0);
    add("FFX_FSR2_OPTION_JITTERED_MOTION_VECTORS",       jitterMV ? 1 : 0);
    add("FFX_FSR2_OPTION_INVERTED_DEPTH",                depthInv ? 1 : 0);
    add("FFX_FSR2_OPTION_APPLY_SHARPENING",              sharpen  ? 1 : 0);
    add("FFX_FSR2_OPTION_REPROJECT_USE_LANCZOS_TYPE",    useLut   ? 1 : 0);
    add("FFX_HALF",                                       fp16     ? 1 : 0);

    return result;
}

// ─── compile + link GLSL compute shader ──────────────────────────────────
static GLuint compileComputeProgram(const std::string& fullSource, std::string* outError = nullptr) {
    auto logError = [&](GLuint obj, bool isShader) {
        GLint len = 0;
        if (isShader) glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &len);
        else          glGetProgramiv(obj, GL_INFO_LOG_LENGTH, &len);
        if (len > 0) {
            std::string msg(len, '\0');
            if (isShader) glGetShaderInfoLog(obj, len, nullptr, msg.data());
            else          glGetProgramInfoLog(obj, len, nullptr, msg.data());
            if (outError) *outError = msg;
        }
    };

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    const char* src = fullSource.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint status = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (!status) {
        logError(shader, true);
        // Debug: dump source for inspection
        static int dump_counter = 0;
        std::string fname = "/tmp/fsr2_fail" + std::to_string(dump_counter++) + ".glsl";
        FILE* f = fopen(fname.c_str(), "w");
        if (f) { fwrite(fullSource.data(), 1, fullSource.size(), f); fclose(f); }
        glDeleteShader(shader);
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (!status) {
        logError(program, false);
        glDeleteShader(shader);
        glDeleteProgram(program);
        return 0;
    }

    glDeleteShader(shader);
    return program;
}

// ─── program introspection for binding layout ────────────────────────────
static void populatePipelineFromProgram(GLuint program, FfxPipelineState* outPipeline) {
    // Clear
    outPipeline->srvCount = 0;
    outPipeline->uavCount = 0;
    outPipeline->constCount = 0;
    outPipeline->rootSignature = nullptr;
    outPipeline->pipeline = reinterpret_cast<FfxPipeline>(static_cast<uintptr_t>(program));

    // Enumerate uniform blocks → constant buffers
    GLint numUBOs = 0;
    glGetProgramiv(program, GL_ACTIVE_UNIFORM_BLOCKS, &numUBOs);
    for (GLint i = 0; i < numUBOs && i < FFX_MAX_NUM_CONST_BUFFERS; i++) {
        GLint binding = 0;
        glGetActiveUniformBlockiv(program, i, GL_UNIFORM_BLOCK_BINDING, &binding);
        char name[64] = {};
        GLsizei len = 0;
        glGetActiveUniformBlockName(program, i, sizeof(name), &len, name);

        auto& cb = outPipeline->cbResourceBindings[outPipeline->constCount];
        cb.slotIndex = static_cast<uint32_t>(binding);
        cb.resourceIdentifier = 0; // patched by patchResourceBindings
        std::mbstowcs(cb.name, name, sizeof(cb.name) / sizeof(wchar_t));
        outPipeline->constCount++;
    }

    // Enumerate active uniforms → SRV (sampler) and UAV (image)
    GLint numUniforms = 0;
    glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &numUniforms);

    for (GLint i = 0; i < numUniforms; i++) {
        char name[64] = {};
        GLsizei len = 0;
        GLenum type = GL_NONE;
        GLint size = 0;
        glGetActiveUniform(program, i, sizeof(name), &len, &size, &type, name);

        // Skip if it's a block member
        GLint blockIdx = -1;
        glGetActiveUniformsiv(program, 1, (const GLuint*)&i, GL_UNIFORM_BLOCK_INDEX, &blockIdx);
        if (blockIdx != -1) continue;

        GLint bindingVal = 0;
        GLint loc = glGetUniformLocation(program, name);
        if (loc != -1) {
            // For sampler/image types, get the binding (initial value from layout(binding=X))
            // We use glGetUniformiv which reads back the default value
            glGetUniformiv(program, loc, &bindingVal);
        }

        bool isSampler = (type == GL_SAMPLER_2D || type == GL_INT_SAMPLER_2D || type == GL_UNSIGNED_INT_SAMPLER_2D ||
                          type == GL_SAMPLER_2D_ARRAY || type == GL_INT_SAMPLER_2D_ARRAY || type == GL_UNSIGNED_INT_SAMPLER_2D_ARRAY ||
                          type == GL_SAMPLER_CUBE || type == GL_SAMPLER_2D_SHADOW || type == GL_SAMPLER_2D_RECT);
        bool isImage = (type == GL_IMAGE_2D || type == GL_INT_IMAGE_2D || type == GL_UNSIGNED_INT_IMAGE_2D ||
                        type == GL_IMAGE_2D_ARRAY || type == GL_INT_IMAGE_2D_ARRAY || type == GL_UNSIGNED_INT_IMAGE_2D_ARRAY);
        // Also handle EXT_samplerless_texture_functions types
        bool isTexture = (type == GL_SAMPLER_2D); // same as sampler, the extension makes texture2D act as a separate type but in GL it's the same enum
        // Actually texture2D maps to GL_SAMPLER_2D in the GL API even with EXT_samplerless_texture_functions
        // image2D maps to GL_IMAGE_2D

        if (isSampler || isTexture) {
            if (outPipeline->srvCount >= FFX_MAX_NUM_SRVS) continue;
            auto& srv = outPipeline->srvResourceBindings[outPipeline->srvCount];
            srv.slotIndex = static_cast<uint32_t>(bindingVal);
            srv.resourceIdentifier = 0;
            std::mbstowcs(srv.name, name, sizeof(srv.name) / sizeof(wchar_t));
            outPipeline->srvCount++;
        } else if (isImage) {
            if (outPipeline->uavCount >= FFX_MAX_NUM_UAVS) continue;
            auto& uav = outPipeline->uavResourceBindings[outPipeline->uavCount];
            uav.slotIndex = static_cast<uint32_t>(bindingVal);
            uav.resourceIdentifier = 0;
            std::mbstowcs(uav.name, name, sizeof(uav.name) / sizeof(wchar_t));
            outPipeline->uavCount++;
        }
    }
}

// ─── format conversion helpers (adapted from fork) ───────────────────────
static GLenum getGLFormatFromSurfaceFormat(FfxSurfaceFormat fmt) {
    switch (fmt) {
        case FFX_SURFACE_FORMAT_R32G32B32A32_TYPELESS:
        case FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT:      return GL_RGBA32F;
        case FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT:      return GL_RGBA16F;
        case FFX_SURFACE_FORMAT_R16G16B16A16_UNORM:      return GL_RGBA16;
        case FFX_SURFACE_FORMAT_R32G32_FLOAT:             return GL_RG32F;
        case FFX_SURFACE_FORMAT_R32_UINT:                 return GL_R32UI;
        case FFX_SURFACE_FORMAT_R8G8B8A8_TYPELESS:
        case FFX_SURFACE_FORMAT_R8G8B8A8_UNORM:           return GL_RGBA8;
        case FFX_SURFACE_FORMAT_R11G11B10_FLOAT:          return GL_R11F_G11F_B10F;
        case FFX_SURFACE_FORMAT_R16G16_FLOAT:             return GL_RG16F;
        case FFX_SURFACE_FORMAT_R16G16_UINT:              return GL_RG16UI;
        case FFX_SURFACE_FORMAT_R16_FLOAT:                return GL_R16F;
        case FFX_SURFACE_FORMAT_R16_UINT:                 return GL_R16UI;
        case FFX_SURFACE_FORMAT_R16_UNORM:                return GL_R16;
        case FFX_SURFACE_FORMAT_R16_SNORM:                return GL_R16_SNORM;
        case FFX_SURFACE_FORMAT_R8_UNORM:                 return GL_R8;
        case FFX_SURFACE_FORMAT_R8G8_UNORM:               return GL_RG8;
        case FFX_SURFACE_FORMAT_R32_FLOAT:                return GL_R32F;
        case FFX_SURFACE_FORMAT_R8_UINT:                  return GL_R8UI;
        default: return 0;
    }
}

static GLenum getGLUploadFormatFromSurfaceFormat(FfxSurfaceFormat fmt) {
    switch (fmt) {
        case FFX_SURFACE_FORMAT_R32G32B32A32_TYPELESS:
        case FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT:
        case FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT:
        case FFX_SURFACE_FORMAT_R16G16B16A16_UNORM:
        case FFX_SURFACE_FORMAT_R8G8B8A8_TYPELESS:
        case FFX_SURFACE_FORMAT_R8G8B8A8_UNORM:          return GL_RGBA;
        case FFX_SURFACE_FORMAT_R11G11B10_FLOAT:          return GL_RGB;
        case FFX_SURFACE_FORMAT_R32G32_FLOAT:
        case FFX_SURFACE_FORMAT_R16G16_FLOAT:
        case FFX_SURFACE_FORMAT_R16G16_UINT:
        case FFX_SURFACE_FORMAT_R8G8_UNORM:               return GL_RG;
        case FFX_SURFACE_FORMAT_R16_FLOAT:
        case FFX_SURFACE_FORMAT_R16_UNORM:
        case FFX_SURFACE_FORMAT_R16_SNORM:
        case FFX_SURFACE_FORMAT_R8_UNORM:
        case FFX_SURFACE_FORMAT_R32_FLOAT:                return GL_RED;
        case FFX_SURFACE_FORMAT_R8_UINT:
        case FFX_SURFACE_FORMAT_R16_UINT:
        case FFX_SURFACE_FORMAT_R32_UINT:                 return GL_RED_INTEGER;
        default: return GL_NONE;
    }
}

static GLenum getGLUploadTypeFromSurfaceFormat(FfxSurfaceFormat fmt) {
    switch (fmt) {
        case FFX_SURFACE_FORMAT_R32G32B32A32_TYPELESS:
        case FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT:
        case FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT:
        case FFX_SURFACE_FORMAT_R32G32_FLOAT:
        case FFX_SURFACE_FORMAT_R11G11B10_FLOAT:
        case FFX_SURFACE_FORMAT_R16G16_FLOAT:
        case FFX_SURFACE_FORMAT_R16_FLOAT:
        case FFX_SURFACE_FORMAT_R32_FLOAT:                return GL_FLOAT;
        case FFX_SURFACE_FORMAT_R8G8B8A8_UNORM:
        case FFX_SURFACE_FORMAT_R8G8B8A8_TYPELESS:
        case FFX_SURFACE_FORMAT_R8G8_UNORM:
        case FFX_SURFACE_FORMAT_R8_UNORM:                 return GL_UNSIGNED_BYTE;
        case FFX_SURFACE_FORMAT_R32_UINT:                 return GL_UNSIGNED_INT;
        case FFX_SURFACE_FORMAT_R16G16B16A16_UNORM:
        case FFX_SURFACE_FORMAT_R16_UNORM:
        case FFX_SURFACE_FORMAT_R16G16_UINT:
        case FFX_SURFACE_FORMAT_R16_UINT:
        case FFX_SURFACE_FORMAT_R8_UINT:                  return GL_UNSIGNED_SHORT;
        case FFX_SURFACE_FORMAT_R16_SNORM:                return GL_SHORT;
        default: return GL_NONE;
    }
}

// ─── callback implementations ────────────────────────────────────────────

static FfxErrorCode GetDeviceCapabilitiesGL(FfxFsr2Interface* backendInterface, FfxDeviceCapabilities* deviceCapabilities, FfxDevice) {
    auto* ctx = static_cast<BackendContext_GL*>(backendInterface->scratchBuffer);
    deviceCapabilities->minimumSupportedShaderModel = FFX_SHADER_MODEL_5_1;
    deviceCapabilities->waveLaneCountMin = 0;
    deviceCapabilities->waveLaneCountMax = 0;
    deviceCapabilities->fp16Supported = false;
    deviceCapabilities->raytracingSupported = false;

    // Check vendor for AMD workaround (subgroup not reported but works)
    const char* vendor = (const char*)glGetString(GL_VENDOR);
    bool vendorIsAmd = vendor && strstr(vendor, "ATI");
    bool subgroupSupported = false;

    GLint numExt = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &numExt);
    for (GLint i = 0; i < numExt; i++) {
        const char* ext = (const char*)glGetStringi(GL_EXTENSIONS, i);
        if (vendorIsAmd || (ext && strcmp(ext, "GL_KHR_shader_subgroup") == 0)) {
            GLint supportedStages = 0;
            glGetIntegerv(GL_SUBGROUP_SUPPORTED_STAGES_KHR, &supportedStages);
            if (supportedStages & GL_COMPUTE_SHADER_BIT)
                subgroupSupported = true;
        }
        if (ext) {
            if (strcmp(ext, "GL_NV_gpu_shader5") == 0 || strcmp(ext, "GL_AMD_gpu_shader_half_float") == 0)
                deviceCapabilities->fp16Supported = true;
        }
    }
    if (!subgroupSupported)
        return FFX_ERROR_BACKEND_API_ERROR;

    GLint subgroupSize = FSR2_DEFAULT_SUBGROUP_SIZE;
    glGetIntegerv(GL_SUBGROUP_SIZE_KHR, &subgroupSize);
    deviceCapabilities->waveLaneCountMin = static_cast<uint32_t>(subgroupSize);
    deviceCapabilities->waveLaneCountMax = static_cast<uint32_t>(subgroupSize);

    return FFX_OK;
}

static FfxErrorCode CreateBackendContextGL(FfxFsr2Interface* backendInterface, FfxDevice) {
    auto* ctx = static_cast<BackendContext_GL*>(backendInterface->scratchBuffer);
    *ctx = {};

    // We need to discover the shaders directory. Use a compile-time define.
    const char* shadersDir = FSR2_SHADERS_DIR;
    ctx->shaderIncludeDirs.push_back(shadersDir);
    // Also include the API root for headers that use relative includes like "ffx_core.h"
    // which are in shaders/ itself — no extra dir needed.

    FFX_VALIDATE(GetDeviceCapabilitiesGL(backendInterface, &ctx->capabilities, nullptr));

    // Create samplers
    glCreateSamplers(1, &ctx->pointSampler.id);
    glSamplerParameteri(ctx->pointSampler.id, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glSamplerParameteri(ctx->pointSampler.id, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glSamplerParameteri(ctx->pointSampler.id, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(ctx->pointSampler.id, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(ctx->pointSampler.id, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glSamplerParameterf(ctx->pointSampler.id, GL_TEXTURE_MIN_LOD, -1000);
    glSamplerParameterf(ctx->pointSampler.id, GL_TEXTURE_MAX_LOD, 1000);
    glSamplerParameterf(ctx->pointSampler.id, GL_TEXTURE_MAX_ANISOTROPY, 1);

    glCreateSamplers(1, &ctx->linearSampler.id);
    glSamplerParameteri(ctx->linearSampler.id, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
    glSamplerParameteri(ctx->linearSampler.id, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(ctx->linearSampler.id, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(ctx->linearSampler.id, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(ctx->linearSampler.id, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glSamplerParameterf(ctx->linearSampler.id, GL_TEXTURE_MIN_LOD, -1000);
    glSamplerParameterf(ctx->linearSampler.id, GL_TEXTURE_MAX_LOD, 1000);
    glSamplerParameterf(ctx->linearSampler.id, GL_TEXTURE_MAX_ANISOTROPY, 1);

    // Allocate UBO ring buffer
    for (uint32_t i = 0; i < FSR2_UBO_RING_BUFFER_SIZE; i++) {
        auto& ubo = ctx->uboRingBuffer[i];
        glCreateBuffers(1, &ubo.bufferResource.id);
        constexpr GLbitfield mapFlags = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
        glNamedBufferStorage(ubo.bufferResource.id, FSR2_UBO_SIZE, nullptr, mapFlags);
        ubo.pData = (uint8_t*)glMapNamedBufferRange(ubo.bufferResource.id, 0, FSR2_UBO_SIZE, mapFlags);
        if (!ubo.pData)
            return FFX_ERROR_BACKEND_API_ERROR;
    }

    ctx->nextStaticResource = 0;
    ctx->nextDynamicResource = FSR2_MAX_RESOURCE_COUNT - 1;
    ctx->gpuJobCount = 0;
    ctx->stagingResourceCount = 0;
    ctx->uboRingBufferIndex = 0;

    return FFX_OK;
}

static FfxErrorCode DestroyBackendContextGL(FfxFsr2Interface* backendInterface) {
    auto* ctx = static_cast<BackendContext_GL*>(backendInterface->scratchBuffer);

    for (uint32_t i = 0; i < ctx->stagingResourceCount; i++) {
        FfxResourceInternal res = ctx->stagingResources[i];
        if (res.internalIndex != -1 && ctx->resources[res.internalIndex].buffer.id)
            glDeleteBuffers(1, &ctx->resources[res.internalIndex].buffer.id);
    }
    for (uint32_t i = 0; i < FSR2_UBO_RING_BUFFER_SIZE; i++) {
        glDeleteBuffers(1, &ctx->uboRingBuffer[i].bufferResource.id);
    }
    glDeleteSamplers(1, &ctx->pointSampler.id);
    glDeleteSamplers(1, &ctx->linearSampler.id);

    *ctx = {};
    return FFX_OK;
}

static FfxErrorCode CreateResourceGL(FfxFsr2Interface* backendInterface, const FfxCreateResourceDescription* desc, FfxResourceInternal* outResource) {
    auto* ctx = static_cast<BackendContext_GL*>(backendInterface->scratchBuffer);
    outResource->internalIndex = ctx->nextStaticResource++;
    auto* res = &ctx->resources[outResource->internalIndex];
    res->resourceDescription = desc->resourceDescription;
    res->resourceDescription.mipCount = desc->resourceDescription.mipCount;
    if (res->resourceDescription.mipCount == 0) {
        uint32_t maxDim = std::max({desc->resourceDescription.width, desc->resourceDescription.height, desc->resourceDescription.depth});
        res->resourceDescription.mipCount = (uint32_t)(1 + floor(log2((float)maxDim)));
    }

    switch (desc->resourceDescription.type) {
    case FFX_RESOURCE_TYPE_BUFFER: {
        glCreateBuffers(1, &res->buffer.id);
        glNamedBufferStorage(res->buffer.id, desc->resourceDescription.width, desc->initData, 0);
        if (desc->name)
            glObjectLabel(GL_BUFFER, res->buffer.id, -1, reinterpret_cast<const char*>(desc->name));
        break;
    }
    case FFX_RESOURCE_TYPE_TEXTURE1D: {
        glCreateTextures(GL_TEXTURE_1D, 1, &res->textureAllMipsView.id);
        glTextureStorage1D(res->textureAllMipsView.id, res->resourceDescription.mipCount,
                             getGLFormatFromSurfaceFormat(desc->resourceDescription.format),
                             desc->resourceDescription.width);
        if (desc->initData) {
            glTextureSubImage1D(res->textureAllMipsView.id, 0, 0, desc->resourceDescription.width,
                                  getGLUploadFormatFromSurfaceFormat(desc->resourceDescription.format),
                                  getGLUploadTypeFromSurfaceFormat(desc->resourceDescription.format),
                                  desc->initData);
        }
        break;
    }
    case FFX_RESOURCE_TYPE_TEXTURE2D: {
        glCreateTextures(GL_TEXTURE_2D, 1, &res->textureAllMipsView.id);
        glTextureStorage2D(res->textureAllMipsView.id, res->resourceDescription.mipCount,
                             getGLFormatFromSurfaceFormat(desc->resourceDescription.format),
                             desc->resourceDescription.width, desc->resourceDescription.height);
        if (desc->initData) {
            glTextureSubImage2D(res->textureAllMipsView.id, 0, 0, 0,
                                  desc->resourceDescription.width, desc->resourceDescription.height,
                                  getGLUploadFormatFromSurfaceFormat(desc->resourceDescription.format),
                                  getGLUploadTypeFromSurfaceFormat(desc->resourceDescription.format),
                                  desc->initData);
        }
        break;
    }
    case FFX_RESOURCE_TYPE_TEXTURE3D: {
        glCreateTextures(GL_TEXTURE_3D, 1, &res->textureAllMipsView.id);
        glTextureStorage3D(res->textureAllMipsView.id, res->resourceDescription.mipCount,
                             getGLFormatFromSurfaceFormat(desc->resourceDescription.format),
                             desc->resourceDescription.width, desc->resourceDescription.height, desc->resourceDescription.depth);
        if (desc->initData) {
            glTextureSubImage3D(res->textureAllMipsView.id, 0, 0, 0, 0,
                                  desc->resourceDescription.width, desc->resourceDescription.height, desc->resourceDescription.depth,
                                  getGLUploadFormatFromSurfaceFormat(desc->resourceDescription.format),
                                  getGLUploadTypeFromSurfaceFormat(desc->resourceDescription.format),
                                  desc->initData);
        }
        break;
    }
    }

    if (desc->resourceDescription.type != FFX_RESOURCE_TYPE_BUFFER) {
        res->textureAspect = BackendContext_GL::Aspect::COLOR;
        GLenum texType = 0;
        switch (desc->resourceDescription.type) {
            case FFX_RESOURCE_TYPE_TEXTURE1D: texType = GL_TEXTURE_1D; break;
            case FFX_RESOURCE_TYPE_TEXTURE2D: texType = GL_TEXTURE_2D; break;
            case FFX_RESOURCE_TYPE_TEXTURE3D: texType = GL_TEXTURE_3D; break;
            default: break;
        }
        for (uint32_t i = 0; i < res->resourceDescription.mipCount && i < FSR2_MAX_IMAGE_VIEWS; i++) {
            glGenTextures(1, &res->textureSingleMipViews[i].id);
            glTextureView(res->textureSingleMipViews[i].id, texType, res->textureAllMipsView.id,
                            getGLFormatFromSurfaceFormat(desc->resourceDescription.format),
                            i, 1, 0, 1);
        }
    }

    return FFX_OK;
}

static FfxErrorCode RegisterResourceGL(FfxFsr2Interface* backendInterface, const FfxResource* inFfxResource, FfxResourceInternal* outFfxResourceInternal) {
    auto* ctx = static_cast<BackendContext_GL*>(backendInterface->scratchBuffer);
    if (!inFfxResource->resource) {
        outFfxResourceInternal->internalIndex = FFX_FSR2_RESOURCE_IDENTIFIER_NULL;
        return FFX_OK;
    }
    outFfxResourceInternal->internalIndex = ctx->nextDynamicResource--;
    auto* backendResource = &ctx->resources[outFfxResourceInternal->internalIndex];
    backendResource->resourceDescription = inFfxResource->description;

    if (inFfxResource->description.type == FFX_RESOURCE_TYPE_BUFFER) {
        backendResource->buffer.id = static_cast<GLuint>(reinterpret_cast<uintptr_t>(inFfxResource->resource));
    } else {
        GLuint tex = static_cast<GLuint>(reinterpret_cast<uintptr_t>(inFfxResource->resource));
        backendResource->textureAllMipsView.id = tex;
        backendResource->textureSingleMipViews[0].id = tex;
        backendResource->textureAspect = inFfxResource->isDepth ? BackendContext_GL::Aspect::DEPTH : BackendContext_GL::Aspect::COLOR;
    }
    return FFX_OK;
}

static FfxErrorCode UnregisterResourcesGL(FfxFsr2Interface* backendInterface) {
    auto* ctx = static_cast<BackendContext_GL*>(backendInterface->scratchBuffer);
    ctx->nextDynamicResource = FSR2_MAX_RESOURCE_COUNT - 1;
    return FFX_OK;
}

static FfxResourceDescription GetResourceDescriptorGL(FfxFsr2Interface* backendInterface, FfxResourceInternal resource) {
    auto* ctx = static_cast<BackendContext_GL*>(backendInterface->scratchBuffer);
    if (resource.internalIndex == -1) return {};
    return ctx->resources[resource.internalIndex].resourceDescription;
}

static FfxErrorCode DestroyResourceGL(FfxFsr2Interface* backendInterface, FfxResourceInternal resource) {
    auto* ctx = static_cast<BackendContext_GL*>(backendInterface->scratchBuffer);
    if (resource.internalIndex == -1) return FFX_OK;
    auto& res = ctx->resources[resource.internalIndex];
    if (res.resourceDescription.type == FFX_RESOURCE_TYPE_BUFFER) {
        if (res.buffer.id) { glDeleteBuffers(1, &res.buffer.id); res.buffer = {}; }
    } else {
        if (res.textureAllMipsView.id) { glDeleteTextures(1, &res.textureAllMipsView.id); res.textureAllMipsView = {}; }
        for (uint32_t i = 0; i < res.resourceDescription.mipCount && i < FSR2_MAX_IMAGE_VIEWS; i++) {
            if (res.textureSingleMipViews[i].id) { glDeleteTextures(1, &res.textureSingleMipViews[i].id); res.textureSingleMipViews[i] = {}; }
        }
    }
    return FFX_OK;
}

static FfxErrorCode CreatePipelineGL(FfxFsr2Interface* backendInterface, FfxFsr2Pass pass, const FfxPipelineDescription* pipelineDescription, FfxPipelineState* outPipeline) {
    auto* ctx = static_cast<BackendContext_GL*>(backendInterface->scratchBuffer);

    const char* filename = passGLSLFile(pass);
    if (!filename) return FFX_ERROR_INVALID_ARGUMENT;

    std::string shaderDir(FSR2_SHADERS_DIR);
    std::string filepath = shaderDir + "/" + filename;

    // Build permutation defines (also returns macro table for include resolver)
    auto perm = buildPermutationDefines(pipelineDescription, pass, ctx->capabilities);

    // Read source, inject #include "ffx_core.h" after #version so types are
    // defined before ffx_fsr2_common.h uses them
    auto rawSrc = gl::read_file(filepath);
    if (!rawSrc.success) {
        GFX_LOG_ERROR("FSR2: failed to read %s", filename);
        return FFX_ERROR_BACKEND_API_ERROR;
    }
    std::string srcText = std::move(rawSrc.source);
    {
        size_t vpos = srcText.find("#version");
        if (vpos != std::string::npos) {
            size_t nl = srcText.find('\n', vpos);
            if (nl != std::string::npos) {
                srcText.insert(nl + 1, "#include \"ffx_core.h\"\n");
            }
        }
        // Also insert #define FFX_GPU and FFX_GLSL at the top so the
        // preprocessor-aware include resolver sees them
        srcText.insert(0, "#define FFX_GPU 1\n#define FFX_GLSL 1\n");
    }

    // Resolve includes with macro-aware preprocessing
    auto includeResult = gl::resolve_includes(filepath, ctx->shaderIncludeDirs, perm.macros);
    if (!includeResult.success) {
        GFX_LOG_ERROR("FSR2: failed to resolve includes for %s: %s", filename, includeResult.error.c_str());
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    std::string& src = includeResult.source;

    // Remove the temporary #define FFX_GPU/FFX_GLSL we prepended (they'll be
    // re-inserted after #version below)
    {
        auto strip_def = [&](const char* name) {
            std::string pat = "#define ";
            pat += name;
            size_t pos = src.find(pat);
            if (pos != std::string::npos) {
                size_t nl = src.find('\n', pos);
                src.erase(pos, (nl == std::string::npos) ? std::string::npos : nl - pos + 1);
            }
        };
        strip_def("FFX_GPU");
        strip_def("FFX_GLSL");
    }

    // Inject permutation defines after #version line
    size_t vpos = src.find("#version");
    size_t insertPos = 0;
    if (vpos != std::string::npos) {
        insertPos = src.find('\n', vpos);
        if (insertPos != std::string::npos) insertPos++;
    }
    src.insert(insertPos, perm.glslText);

    // Collect ALL #extension lines, remove from current positions, filter
    // unsupported ones, then re-insert supported ones at top. This is needed
    // because included headers (e.g. ffx_spd.h) may declare #extension deep
    // in the file after non-preprocessor tokens, which GLSL prohibits.
    {
        const char* unsupported[] = {
            "GL_GOOGLE_include_directive",
            "GL_EXT_samplerless_texture_functions",
            "GL_EXT_shader_image_load_formatted",
        };
        auto isUnsupported = [&](const std::string& ext) {
            for (auto u : unsupported)
                if (ext == u) return true;
            return false;
        };

        // Collect all #extension lines
        struct ExtLine { size_t start, end; std::string name; };
        std::vector<ExtLine> extLines;
        size_t pos = 0;
        while (pos < src.size()) {
            size_t nl = src.find('#', pos);
            if (nl == std::string::npos) break;
            size_t line_start = nl;
            size_t line_end = src.find('\n', nl);
            line_end = (line_end == std::string::npos) ? src.size() : line_end + 1;
            // Check if this is an #extension directive
            if (src.compare(nl, 10, "#extension") == 0) {
                // Extract extension name
                size_t ext_start = nl + 10;
                // Skip whitespace and ':'
                while (ext_start < line_end && (src[ext_start] == ' ' || src[ext_start] == '\t')) ext_start++;
                size_t ext_end = src.find_first_of(" :;\n", ext_start);
                if (ext_end != std::string::npos) {
                    std::string extName = src.substr(ext_start, ext_end - ext_start);
                    // Strip any leading/trailing whitespace
                    extName.erase(0, extName.find_first_not_of(" \t"));
                    extName.erase(extName.find_last_not_of(" \t") + 1);
                    extLines.push_back({line_start, line_end, extName});
                }
            }
            pos = line_end;
        }

        // Remove all #extension lines from their current positions (reversed to keep offsets valid)
        for (auto it = extLines.rbegin(); it != extLines.rend(); ++it)
            src.erase(it->start, it->end - it->start);

        // Insert supported extensions at the top, after #version + permutation defines
        size_t insertPoint = src.find('\n', src.find("#version"));
        insertPoint = (insertPoint == std::string::npos) ? 0 : insertPoint + 1;
        // Find end of permutation defines (they end with a blank line typically)
        // Just insert after the #version line
        std::string extBlock;
        for (auto& el : extLines) {
            if (!isUnsupported(el.name))
                extBlock += "#extension " + el.name + " : require\n";
        }
        src.insert(insertPoint, extBlock);
    }

    // Compile
    std::string compileError;
    GLuint program = compileComputeProgram(src, &compileError);
    if (!program) {
        GFX_LOG_ERROR("FSR2: failed to compile %s:\n%s", filename, compileError.c_str());
        GFX_LOG_ERROR("FSR2: full source:\n%s", src.c_str());
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    // Populate pipeline binding info from program introspection
    populatePipelineFromProgram(program, outPipeline);

    return FFX_OK;
}

static FfxErrorCode DestroyPipelineGL(FfxFsr2Interface* backendInterface, FfxPipelineState* pipeline) {
    if (!pipeline) return FFX_OK;
    GLuint program = static_cast<GLuint>(reinterpret_cast<uintptr_t>(pipeline->pipeline));
    if (program) glDeleteProgram(program);
    return FFX_OK;
}

static BackendContext_GL::UniformBuffer acquireDynamicUBO(BackendContext_GL* ctx, uint32_t size, const void* pData) {
    auto& ubo = ctx->uboRingBuffer[ctx->uboRingBufferIndex];
    if (pData) memcpy(ubo.pData, pData, size);
    ctx->uboRingBufferIndex = (ctx->uboRingBufferIndex + 1) % FSR2_UBO_RING_BUFFER_SIZE;
    return ubo;
}

static void addBarrier(bool isBufferBarrier, FfxResourceStates newState) {
    GLbitfield barriers = 0;
    if (isBufferBarrier) {
        barriers |= (newState & FFX_RESOURCE_STATE_UNORDERED_ACCESS) ? GL_SHADER_STORAGE_BARRIER_BIT : 0;
        barriers |= (newState & FFX_RESOURCE_STATE_COMPUTE_READ) ? GL_UNIFORM_BARRIER_BIT : 0;
        barriers |= (newState & FFX_RESOURCE_STATE_COPY_SRC) ? (GL_BUFFER_UPDATE_BARRIER_BIT | GL_PIXEL_BUFFER_BARRIER_BIT) : 0;
        barriers |= (newState & FFX_RESOURCE_STATE_COPY_DEST) ? (GL_BUFFER_UPDATE_BARRIER_BIT | GL_PIXEL_BUFFER_BARRIER_BIT) : 0;
    } else {
        barriers |= (newState & FFX_RESOURCE_STATE_UNORDERED_ACCESS) ? GL_SHADER_IMAGE_ACCESS_BARRIER_BIT : 0;
        barriers |= (newState & FFX_RESOURCE_STATE_COMPUTE_READ) ? (GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT) : 0;
        barriers |= (newState & FFX_RESOURCE_STATE_COPY_SRC) ? GL_TEXTURE_UPDATE_BARRIER_BIT : 0;
        barriers |= (newState & FFX_RESOURCE_STATE_COPY_DEST) ? GL_TEXTURE_UPDATE_BARRIER_BIT : 0;
    }
    if (barriers) glMemoryBarrier(barriers);
}

static FfxErrorCode executeGpuJobCompute(BackendContext_GL* ctx, const FfxGpuJobDescription* job) {
    const auto program = static_cast<GLuint>(reinterpret_cast<uintptr_t>(job->computeJobDescriptor.pipeline.pipeline));

    if (job->computeJobDescriptor.pipeline.uavCount > 0)
        addBarrier(false, FFX_RESOURCE_STATE_UNORDERED_ACCESS);

    for (uint32_t uav = 0; uav < job->computeJobDescriptor.pipeline.uavCount; ++uav) {
        auto& ffxRes = ctx->resources[job->computeJobDescriptor.uavs[uav].internalIndex];
        if (!ffxRes.textureSingleMipViews[job->computeJobDescriptor.uavMip[uav]].id) continue;
        glBindImageTexture(
            job->computeJobDescriptor.pipeline.uavResourceBindings[uav].slotIndex,
            ffxRes.textureSingleMipViews[job->computeJobDescriptor.uavMip[uav]].id,
            0, true, 0, GL_READ_WRITE,
            getGLFormatFromSurfaceFormat(ffxRes.resourceDescription.format));
    }

    if (job->computeJobDescriptor.pipeline.srvCount > 0)
        addBarrier(false, FFX_RESOURCE_STATE_COMPUTE_READ);

    for (uint32_t srv = 0; srv < job->computeJobDescriptor.pipeline.srvCount; ++srv) {
        auto& ffxRes = ctx->resources[job->computeJobDescriptor.srvs[srv].internalIndex];
        if (!ffxRes.textureAllMipsView.id) continue;
        glBindTextureUnit(job->computeJobDescriptor.pipeline.srvResourceBindings[srv].slotIndex, ffxRes.textureAllMipsView.id);
        glBindSampler(job->computeJobDescriptor.pipeline.srvResourceBindings[srv].slotIndex, ctx->linearSampler.id);
    }

    for (uint32_t i = 0; i < job->computeJobDescriptor.pipeline.constCount; ++i) {
        auto ubo = acquireDynamicUBO(ctx, job->computeJobDescriptor.cbs[i].uint32Size * sizeof(uint32_t), job->computeJobDescriptor.cbs[i].data);
        glBindBufferRange(GL_UNIFORM_BUFFER,
                            job->computeJobDescriptor.pipeline.cbResourceBindings[i].slotIndex,
                            ubo.bufferResource.id, 0, FSR2_UBO_SIZE);
    }

    glUseProgram(program);
    glDispatchCompute(job->computeJobDescriptor.dimensions[0],
                        job->computeJobDescriptor.dimensions[1],
                        job->computeJobDescriptor.dimensions[2]);
    return FFX_OK;
}

static FfxErrorCode executeGpuJobClearFloat(BackendContext_GL* ctx, const FfxGpuJobDescription* job) {
    uint32_t idx = job->clearJobDescriptor.target.internalIndex;
    if (idx >= FSR2_MAX_RESOURCE_COUNT) return FFX_OK;
    auto& ffxRes = ctx->resources[idx];
    if (ffxRes.resourceDescription.type == FFX_RESOURCE_TYPE_BUFFER)
        return FFX_OK;

    addBarrier(false, FFX_RESOURCE_STATE_COPY_DEST);
    float clearColor[4] = {
        job->clearJobDescriptor.color[0],
        job->clearJobDescriptor.color[1],
        job->clearJobDescriptor.color[2],
        job->clearJobDescriptor.color[3]
    };
    for (uint32_t m = 0; m < ffxRes.resourceDescription.mipCount; m++) {
        glClearTexImage(ffxRes.textureAllMipsView.id, m, GL_RGBA, GL_FLOAT, clearColor);
    }
    return FFX_OK;
}

static FfxErrorCode ScheduleGpuJobGL(FfxFsr2Interface* backendInterface, const FfxGpuJobDescription* job) {
    auto* ctx = static_cast<BackendContext_GL*>(backendInterface->scratchBuffer);
    ctx->gpuJobs[ctx->gpuJobCount] = *job;
    if (job->jobType == FFX_GPU_JOB_COMPUTE) {
        auto* computeJob = &ctx->gpuJobs[ctx->gpuJobCount].computeJobDescriptor;
        for (uint32_t i = 0; i < job->computeJobDescriptor.pipeline.constCount; ++i) {
            computeJob->cbs[i].uint32Size = job->computeJobDescriptor.cbs[i].uint32Size;
            memcpy(computeJob->cbs[i].data, job->computeJobDescriptor.cbs[i].data, computeJob->cbs[i].uint32Size * sizeof(uint32_t));
        }
    }
    ctx->gpuJobCount++;
    return FFX_OK;
}

static FfxErrorCode ExecuteGpuJobsGL(FfxFsr2Interface* backendInterface, FfxCommandList) {
    auto* ctx = static_cast<BackendContext_GL*>(backendInterface->scratchBuffer);
    for (uint32_t i = 0; i < ctx->gpuJobCount; ++i) {
        auto* job = &ctx->gpuJobs[i];
        switch (job->jobType) {
        case FFX_GPU_JOB_CLEAR_FLOAT:
            executeGpuJobClearFloat(ctx, job);
            break;
        case FFX_GPU_JOB_COPY:
            GFX_LOG_WARN("FSR2: copy job not implemented");
            break;
        case FFX_GPU_JOB_COMPUTE:
            executeGpuJobCompute(ctx, job);
            break;
        }
    }
    ctx->gpuJobCount = 0;
    return FFX_OK;
}

} // anonymous namespace

// ─── public API ──────────────────────────────────────────────────────────

size_t ffxFsr2GetScratchMemorySizeGL() {
    return sizeof(BackendContext_GL);
}

FfxErrorCode ffxFsr2GetInterfaceGL(FfxFsr2Interface* outInterface, void* scratchBuffer, size_t scratchBufferSize) {
    if (!outInterface) return FFX_ERROR_INVALID_POINTER;
    if (!scratchBuffer) return FFX_ERROR_INVALID_POINTER;
    if (scratchBufferSize < ffxFsr2GetScratchMemorySizeGL()) return FFX_ERROR_INSUFFICIENT_MEMORY;

    outInterface->fpGetDeviceCapabilities = GetDeviceCapabilitiesGL;
    outInterface->fpCreateBackendContext = CreateBackendContextGL;
    outInterface->fpDestroyBackendContext = DestroyBackendContextGL;
    outInterface->fpCreateResource = CreateResourceGL;
    outInterface->fpRegisterResource = RegisterResourceGL;
    outInterface->fpUnregisterResources = UnregisterResourcesGL;
    outInterface->fpGetResourceDescription = GetResourceDescriptorGL;
    outInterface->fpDestroyResource = DestroyResourceGL;
    outInterface->fpCreatePipeline = CreatePipelineGL;
    outInterface->fpDestroyPipeline = DestroyPipelineGL;
    outInterface->fpScheduleGpuJob = ScheduleGpuJobGL;
    outInterface->fpExecuteGpuJobs = ExecuteGpuJobsGL;
    outInterface->scratchBuffer = scratchBuffer;
    outInterface->scratchBufferSize = scratchBufferSize;

    return FFX_OK;
}

FfxResource ffxGetTextureResourceGL(GLuint textureGL, uint32_t width, uint32_t height, GLenum imgFormat, const wchar_t* name) {
    FfxResource res = {};
    res.resource = reinterpret_cast<void*>(static_cast<uintptr_t>(textureGL));
    res.description.flags = FFX_RESOURCE_FLAGS_NONE;
    res.description.type = FFX_RESOURCE_TYPE_TEXTURE2D;
    res.description.width = width;
    res.description.height = height;
    res.description.depth = 1;
    res.description.mipCount = 1;
    res.description.format = ffxGetSurfaceFormatGL(imgFormat);
    switch (imgFormat) {
        case GL_DEPTH_COMPONENT16: case GL_DEPTH_COMPONENT24: case GL_DEPTH_COMPONENT32F:
        case GL_DEPTH24_STENCIL8:  case GL_DEPTH32F_STENCIL8:
            res.isDepth = true; break;
        default: res.isDepth = false; break;
    }
    if (name) wcscpy(res.name, name);
    return res;
}

FfxResource ffxGetBufferResourceGL(GLuint bufferGL, uint32_t size, const wchar_t* name) {
    FfxResource res = {};
    res.resource = reinterpret_cast<void*>(static_cast<uintptr_t>(bufferGL));
    res.description.flags = FFX_RESOURCE_FLAGS_NONE;
    res.description.type = FFX_RESOURCE_TYPE_BUFFER;
    res.description.width = size;
    res.description.height = 1;
    res.description.depth = 1;
    res.description.mipCount = 1;
    res.description.format = FFX_SURFACE_FORMAT_UNKNOWN;
    res.isDepth = false;
    if (name) wcscpy(res.name, name);
    return res;
}

GLuint ffxGetGLImage(FfxFsr2Context* context, uint32_t resId) {
    auto* ctxPrivate = reinterpret_cast<FfxFsr2Context_Private*>(context);
    auto* ctx = static_cast<BackendContext_GL*>(ctxPrivate->contextDescription.callbacks.scratchBuffer);
    int32_t idx = ctxPrivate->uavResources[resId].internalIndex;
    return (idx == -1) ? 0 : ctx->resources[idx].textureAllMipsView.id;
}

FfxSurfaceFormat ffxGetSurfaceFormatGL(GLenum fmt) {
    switch (fmt) {
        case GL_RGBA32F:        return FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT;
        case GL_RGBA16F:        return FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT;
        case GL_RGBA16:         return FFX_SURFACE_FORMAT_R16G16B16A16_UNORM;
        case GL_RG32F:          return FFX_SURFACE_FORMAT_R32G32_FLOAT;
        case GL_R32UI:          return FFX_SURFACE_FORMAT_R32_UINT;
        case GL_RGBA8:          return FFX_SURFACE_FORMAT_R8G8B8A8_UNORM;
        case GL_R11F_G11F_B10F: return FFX_SURFACE_FORMAT_R11G11B10_FLOAT;
        case GL_RG16F:          return FFX_SURFACE_FORMAT_R16G16_FLOAT;
        case GL_RG16UI:         return FFX_SURFACE_FORMAT_R16G16_UINT;
        case GL_R16F:           return FFX_SURFACE_FORMAT_R16_FLOAT;
        case GL_R16UI:          return FFX_SURFACE_FORMAT_R16_UINT;
        case GL_R16:            return FFX_SURFACE_FORMAT_R16_UNORM;
        case GL_R16_SNORM:      return FFX_SURFACE_FORMAT_R16_SNORM;
        case GL_R8:             return FFX_SURFACE_FORMAT_R8_UNORM;
        case GL_R32F:           return FFX_SURFACE_FORMAT_R32_FLOAT;
        case GL_R8UI:           return FFX_SURFACE_FORMAT_R8_UINT;
        case GL_RG8:            return FFX_SURFACE_FORMAT_R8G8_UNORM;
        default:                return FFX_SURFACE_FORMAT_UNKNOWN;
    }
}
