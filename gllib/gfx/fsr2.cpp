#include "fsr2.hpp"
#include "fsr2_backend_gl.hpp"

#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace gfx {

Fsr2::Fsr2() = default;

Fsr2::~Fsr2() {
    if (initialized_) shutdown();
}

bool Fsr2::init(const Fsr2InitParams& params) {
    if (initialized_) shutdown();

    currentParams_ = params;

    size_t scratchSize = ffxFsr2GetScratchMemorySizeGL();
    scratchBuffer_ = calloc(1, scratchSize);
    if (!scratchBuffer_) {
        std::fprintf(stderr, "Fsr2: failed to allocate scratch buffer (%zu bytes)\n", scratchSize);
        return false;
    }

    FfxFsr2Interface callbacks = {};
    FfxErrorCode err = ffxFsr2GetInterfaceGL(&callbacks, scratchBuffer_, scratchSize);
    if (err != FFX_OK) {
        std::fprintf(stderr, "Fsr2: ffxFsr2GetInterfaceGL failed: %d\n", err);
        free(scratchBuffer_);
        scratchBuffer_ = nullptr;
        return false;
    }

    FfxFsr2ContextDescription ctxDesc = {};
    ctxDesc.flags = params.flags;
    ctxDesc.maxRenderSize.width  = params.renderWidth;
    ctxDesc.maxRenderSize.height = params.renderHeight;
    ctxDesc.displaySize.width    = params.displayWidth;
    ctxDesc.displaySize.height   = params.displayHeight;
    ctxDesc.callbacks = callbacks;
    ctxDesc.device = nullptr;
    ctxDesc.fpMessage = nullptr;

    err = ffxFsr2ContextCreate(&context_, &ctxDesc);
    if (err != FFX_OK) {
        std::fprintf(stderr, "Fsr2: ffxFsr2ContextCreate failed: %d\n", err);
        free(scratchBuffer_);
        scratchBuffer_ = nullptr;
        return false;
    }

    initialized_ = true;
    return true;
}

void Fsr2::dispatch(const Fsr2DispatchParams& params) {
    if (!initialized_) return;

    FfxResource color = ffxGetTextureResourceGL(
        params.color, currentParams_.renderWidth, currentParams_.renderHeight,
        GL_RGBA16F, L"FSR2_Color");
    FfxResource depth = ffxGetTextureResourceGL(
        params.depth, currentParams_.renderWidth, currentParams_.renderHeight,
        GL_DEPTH_COMPONENT32F, L"FSR2_Depth");
    FfxResource motion = ffxGetTextureResourceGL(
        params.motionVectors, currentParams_.renderWidth, currentParams_.renderHeight,
        GL_RG16F, L"FSR2_MotionVectors");
    FfxResource output = ffxGetTextureResourceGL(
        params.output, currentParams_.displayWidth, currentParams_.displayHeight,
        GL_RGBA16F, L"FSR2_Output");

    FfxFloatCoords2D jitter = { params.jitterX, params.jitterY };

    FfxFsr2DispatchDescription dispatchDesc = {};
    dispatchDesc.commandList = nullptr;
    dispatchDesc.color = color;
    dispatchDesc.depth = depth;
    dispatchDesc.motionVectors = motion;
    dispatchDesc.output = output;
    dispatchDesc.jitterOffset = jitter;
    dispatchDesc.motionVectorScale.x = static_cast<float>(currentParams_.renderWidth);
    dispatchDesc.motionVectorScale.y = static_cast<float>(currentParams_.renderHeight);
    dispatchDesc.renderSize.width  = currentParams_.renderWidth;
    dispatchDesc.renderSize.height = currentParams_.renderHeight;
    dispatchDesc.enableSharpening = true;
    dispatchDesc.sharpness = params.sharpness;
    dispatchDesc.frameTimeDelta = params.deltaTime;
    dispatchDesc.preExposure = 1.0f;
    dispatchDesc.reset = params.reset;
    dispatchDesc.cameraNear = params.nearPlane;
    dispatchDesc.cameraFar = params.farPlane;
    dispatchDesc.cameraFovAngleVertical = params.fovVerticalRad;
    dispatchDesc.viewSpaceToMetersFactor = 1.0f;
    dispatchDesc.deviceDepthNegativeOneToOne = true;

    FfxErrorCode err = ffxFsr2ContextDispatch(&context_, &dispatchDesc);
    if (err != FFX_OK) {
        std::fprintf(stderr, "Fsr2: ffxFsr2ContextDispatch failed: %d\n", err);
    }
}

void Fsr2::shutdown() {
    if (!initialized_) return;
    ffxFsr2ContextDestroy(&context_);
    if (scratchBuffer_) {
        free(scratchBuffer_);
        scratchBuffer_ = nullptr;
    }
    initialized_ = false;
}

void Fsr2::on_resize(const Fsr2InitParams& params) {
    if (initialized_) {
        shutdown();
    }
    init(params);
}

} // namespace gfx
