#include "ibl_probe.hpp"
#include "cubemap.hpp"
#include "../gl/shader.hpp"
#include "../gl/program.hpp"
#include "../gl/texture.hpp"
#include <glm/glm.hpp>
#include <cmath>

// Irradiance convolution compute shader
static const char* irradiance_comp = R"(
#version 430 core
layout(local_size_x = 8, local_size_y = 8, local_size_z = 6) in;
layout(binding = 1, rgba16f) uniform imageCube u_out;
uniform samplerCube u_env;
uniform int u_size;
const float PI = 3.14159265;
float radical_inverse(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}
vec2 hammersley(uint i, uint N) { return vec2(float(i)/float(N), radical_inverse(i)); }
vec3 sample_cos(vec2 xi) {
    float phi = 2.0*PI*xi.x;
    float cos_t = sqrt(1.0 - xi.y);
    float sin_t = sqrt(xi.y);
    return vec3(sin_t*cos(phi), sin_t*sin(phi), cos_t);
}
void main() {
    ivec3 coord = ivec3(gl_GlobalInvocationID);
    ivec2 sz = imageSize(u_out).xy;
    if (coord.x >= sz.x || coord.y >= sz.y) return;
    int face = coord.z;
    vec3 dirs[6] = vec3[](vec3(1,0,0),vec3(-1,0,0),vec3(0,1,0),vec3(0,-1,0),vec3(0,0,1),vec3(0,0,-1));
    vec3 ups[6] = vec3[](vec3(0,-1,0),vec3(0,-1,0),vec3(0,0,1),vec3(0,0,-1),vec3(0,-1,0),vec3(0,-1,0));
    vec3 f = dirs[face], u = ups[face], r = normalize(cross(u, f));
    vec2 uv = (vec2(coord.xy)+0.5)/vec2(sz);
    vec3 N = normalize(f + r*(2.0*uv.x-1.0) + u*(2.0*uv.y-1.0));
    vec3 T = normalize(cross(N, abs(N.y)<0.999?vec3(0,1,0):vec3(1,0,0)));
    vec3 B = cross(N, T);
    uint NS = 8192u;
    vec3 irr = vec3(0.0);
    for (uint i = 0u; i < NS; i++) {
        vec2 xi = hammersley(i, NS);
        vec3 L = T*sample_cos(xi).x + B*sample_cos(xi).y + N*sample_cos(xi).z;
        irr += texture(u_env, L).rgb * max(dot(N,L), 0.0);
    }
    irr *= PI/float(NS);
    imageStore(u_out, coord, vec4(irr,1.0));
}
)";

// Specular prefilter compute shader
static const char* prefilter_comp = R"(
#version 430 core
layout(local_size_x = 8, local_size_y = 8, local_size_z = 6) in;
layout(binding = 1, rgba16f) uniform imageCube u_out;
uniform samplerCube u_env;
uniform int u_size;
uniform float u_roughness;
const float PI = 3.14159265;
float radical_inverse(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}
vec2 hammersley(uint i, uint N) { return vec2(float(i)/float(N), radical_inverse(i)); }
vec3 importance_sample_ggx(vec2 xi, vec3 N, float a) {
    float phi = 2.0*PI*xi.x;
    float cos_t = sqrt((1.0-xi.y)/(1.0+(a*a-1.0)*xi.y));
    float sin_t = sqrt(1.0-cos_t*cos_t);
    vec3 H = vec3(sin_t*cos(phi), sin_t*sin(phi), cos_t);
    vec3 up = abs(N.y)<0.999?vec3(0,1,0):vec3(1,0,0);
    vec3 T = normalize(cross(up,N));
    vec3 B = cross(N,T);
    return normalize(T*H.x + B*H.y + N*H.z);
}
void main() {
    ivec3 coord = ivec3(gl_GlobalInvocationID);
    ivec2 sz = imageSize(u_out).xy;
    if (coord.x >= sz.x || coord.y >= sz.y) return;
    int face = coord.z;
    vec3 dirs[6] = vec3[](vec3(1,0,0),vec3(-1,0,0),vec3(0,1,0),vec3(0,-1,0),vec3(0,0,1),vec3(0,0,-1));
    vec3 ups[6] = vec3[](vec3(0,-1,0),vec3(0,-1,0),vec3(0,0,1),vec3(0,0,-1),vec3(0,-1,0),vec3(0,-1,0));
    vec3 f = dirs[face], u = ups[face], r = normalize(cross(u, f));
    vec2 uv = (vec2(coord.xy)+0.5)/vec2(sz);
    vec3 N = normalize(f + r*(2.0*uv.x-1.0) + u*(2.0*uv.y-1.0));
    vec3 V = N;
    float a = u_roughness*u_roughness;
    uint NS = 512u;
    vec3 color = vec3(0.0);
    float total = 0.0;
    for (uint i = 0u; i < NS; i++) {
        vec2 xi = hammersley(i, NS);
        vec3 H = importance_sample_ggx(xi, N, a);
        vec3 L = normalize(2.0*dot(V,H)*H - V);
        float NdotL = max(dot(N,L), 0.0);
        if (NdotL > 0.0) {
            color += texture(u_env, L).rgb * NdotL;
            total += NdotL;
        }
    }
    color = total > 0.0 ? color/total : vec3(0.0);
    imageStore(u_out, coord, vec4(color, 1.0));
}
)";

// BRDF LUT compute shader
static const char* brdf_comp = R"(
#version 430 core
layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 0, rg16f) uniform image2D u_out;
uniform int u_sample_count;
const float PI = 3.14159265;
float radical_inverse(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}
vec2 hammersley(uint i, uint N) { return vec2(float(i)/float(N), radical_inverse(i)); }
vec3 importance_sample_ggx(vec2 xi, vec3 N, float a) {
    float phi = 2.0*PI*xi.x;
    float cos_t = sqrt((1.0-xi.y)/(1.0+(a*a-1.0)*xi.y));
    float sin_t = sqrt(1.0-cos_t*cos_t);
    vec3 H = vec3(sin_t*cos(phi), sin_t*sin(phi), cos_t);
    vec3 up = abs(N.y)<0.999?vec3(0,1,0):vec3(1,0,0);
    vec3 T = normalize(cross(up,N));
    vec3 B = cross(N,T);
    return normalize(T*H.x + B*H.y + N*H.z);
}
void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 sz = imageSize(u_out);
    if (coord.x >= sz.x || coord.y >= sz.y) return;
    float NdotV = (float(coord.x)+0.5)/float(sz.x);
    float rough = (float(coord.y)+0.5)/float(sz.y);
    vec3 V = vec3(sqrt(1.0-NdotV*NdotV), 0.0, NdotV);
    vec3 N = vec3(0.0,0.0,1.0);
    float scale = 0.0, bias = 0.0;
    int S = u_sample_count;
    for (int i = 0; i < S; i++) {
        vec2 xi = hammersley(uint(i), uint(S));
        vec3 H = importance_sample_ggx(xi, N, rough);
        vec3 L = normalize(2.0*dot(V,H)*H - V);
        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V,H), 0.0);
        if (NdotL > 0.0) {
            float a = rough*rough;
            float G1 = 1.0/(1.0+sqrt(1.0+a*a*(1.0-NdotL*NdotL)/(NdotL*NdotL)));
            float G2 = G1*(1.0/(1.0+sqrt(1.0+a*a*(1.0-NdotV*NdotV)/(NdotV*NdotV))));
            float Fc = pow(1.0-VdotH,5.0);
            scale += G2*(1.0-Fc);
            bias += G2*Fc;
        }
    }
    imageStore(u_out, coord, vec4(scale/S, bias/S, 0.0, 0.0));
}
)";

gfx::IBLProbe::IBLProbe() = default;
gfx::IBLProbe::~IBLProbe() { delete env_map_; delete irradiance_map_; delete prefilter_map_; delete brdf_lut_; }

gfx::IBLProbe::IBLProbe(IBLProbe&& other) noexcept
    : env_map_(other.env_map_), irradiance_map_(other.irradiance_map_),
      prefilter_map_(other.prefilter_map_), brdf_lut_(other.brdf_lut_)
{
    other.env_map_ = other.irradiance_map_ = other.prefilter_map_ = nullptr;
    other.brdf_lut_ = nullptr;
}

gfx::IBLProbe& gfx::IBLProbe::operator=(IBLProbe&& other) noexcept {
    if (this != &other) {
        delete env_map_; delete irradiance_map_; delete prefilter_map_; delete brdf_lut_;
        env_map_ = other.env_map_; irradiance_map_ = other.irradiance_map_;
        prefilter_map_ = other.prefilter_map_; brdf_lut_ = other.brdf_lut_;
        other.env_map_ = other.irradiance_map_ = other.prefilter_map_ = nullptr;
        other.brdf_lut_ = nullptr;
    }
    return *this;
}

void gfx::IBLProbe::generate_procedural(int size) {
    delete env_map_;
    env_map_ = new Cubemap;
    env_map_->generate_procedural_hdr(size);

    irradiance_map_ = new Cubemap;
    irradiance_map_->create_storage(64, 1, GL_RGBA16F);

    int prefilter_size = size / 2;
    prefilter_map_ = new Cubemap;
    prefilter_map_->create_storage(prefilter_size, 5, GL_RGBA16F);

    const int lut_size = 256;
    brdf_lut_ = new gl::Texture(gl::TextureType::tex_2d);
    brdf_lut_->bind(0);
    brdf_lut_->image_2d(0, GL_RG16F, lut_size, lut_size, GL_RG, GL_FLOAT, nullptr, 1);
    brdf_lut_->parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    brdf_lut_->parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    brdf_lut_->parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    brdf_lut_->parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void gfx::IBLProbe::bake() {
    if (!env_map_ || !irradiance_map_ || !prefilter_map_ || !brdf_lut_) return;

    // Bind env map as sampler (texture unit 0, used by compute shaders)
    env_map_->bind(0);

    // --- Irradiance ---
    {
        gl::Shader cs(gl::ShaderType::compute, irradiance_comp);
        gl::Program prog;
        if (!cs.compiled()) return;
        prog.attach(cs);
        if (!prog.link()) return;
        prog.use();
        irradiance_map_->bind_image_layered(0, 1, GL_WRITE_ONLY, GL_RGBA16F);
        GLint loc = prog.uniform_location("u_size");
        if (loc >= 0) prog.uniform1i(loc, irradiance_map_->width());
        int gx = (irradiance_map_->width() + 7) / 8;
        int gy = (irradiance_map_->height() + 7) / 8;
        glDispatchCompute(gx, gy, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
    }

    // --- Prefilter ---
    {
        gl::Shader cs(gl::ShaderType::compute, prefilter_comp);
        gl::Program prog;
        if (!cs.compiled()) return;
        prog.attach(cs);
        if (!prog.link()) return;
        prog.use();
        GLint loc_size = prog.uniform_location("u_size");
        if (loc_size >= 0) prog.uniform1i(loc_size, env_map_->width());
        for (int mip = 0; mip < prefilter_map_->levels(); mip++) {
            int mip_w = std::max(prefilter_map_->width() >> mip, 1);
            int mip_h = std::max(prefilter_map_->height() >> mip, 1);
            prefilter_map_->bind_image_layered(mip, 1, GL_WRITE_ONLY, GL_RGBA16F);
            float roughness = float(mip) / float(prefilter_map_->levels() - 1);
            GLint loc_rough = prog.uniform_location("u_roughness");
            if (loc_rough >= 0) prog.uniform1f(loc_rough, roughness);
            int gx = (mip_w + 7) / 8;
            int gy = (mip_h + 7) / 8;
            glDispatchCompute(gx, gy, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        }
    }

    // --- BRDF LUT ---
    {
        gl::Shader cs(gl::ShaderType::compute, brdf_comp);
        gl::Program prog;
        if (!cs.compiled()) return;
        prog.attach(cs);
        if (!prog.link()) return;
        prog.use();
        brdf_lut_->bind_image(0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG16F);
        GLint loc = prog.uniform_location("u_sample_count");
        if (loc >= 0) prog.uniform1i(loc, 1024);
        int gx = (256 + 15) / 16;
        int gy = (256 + 15) / 16;
        glDispatchCompute(gx, gy, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }
}
