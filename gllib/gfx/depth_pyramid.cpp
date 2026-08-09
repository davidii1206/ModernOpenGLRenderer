#include "depth_pyramid.hpp"

#include <gl/shader.hpp>
#include <gllib/log.hpp>

#include <cstdio>
#include <vector>

namespace gfx {

// Copy depth texture → level 0 of Hi-Z (R32F)
static const char* copy_comp_src = R"(
#version 460 core
layout(local_size_x = 16, local_size_y = 16) in;

layout(binding=0) uniform sampler2D u_depth;
layout(r32f, binding=1) uniform image2D u_hiz;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 sz = imageSize(u_hiz);
    if (coord.x >= sz.x || coord.y >= sz.y) return;
    float d = texelFetch(u_depth, coord, 0).r;
    imageStore(u_hiz, coord, vec4(d));
}
)";

// Reduce one Hi-Z mip level → next (max of 2x2)
static const char* reduce_comp_src = R"(
#version 460 core
layout(local_size_x = 16, local_size_y = 16) in;

layout(r32f, binding=0) uniform image2D u_input;
layout(r32f, binding=1) uniform image2D u_output;

void main() {
    ivec2 base = ivec2(gl_GlobalInvocationID.xy) * 2;
    ivec2 out_sz = imageSize(u_output);
    if (gl_GlobalInvocationID.x >= out_sz.x || gl_GlobalInvocationID.y >= out_sz.y) return;

    float d0 = imageLoad(u_input, base).r;
    float d1 = imageLoad(u_input, base + ivec2(1, 0)).r;
    float d2 = imageLoad(u_input, base + ivec2(0, 1)).r;
    float d3 = imageLoad(u_input, base + ivec2(1, 1)).r;

    imageStore(u_output, ivec2(gl_GlobalInvocationID.xy),
               vec4(max(max(d0, d1), max(d2, d3))));
}
)";

static GLuint compile_compute(const char* src, const char* name) {
    gl::Shader cs(gl::ShaderType::compute, src);
    if (!cs.compiled()) return 0;
    GLuint prog = glCreateProgram();
    glAttachShader(prog, cs.handle());
    glLinkProgram(prog);
    GLint status = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &status);
    if (!status) {
        char buf[512];
        GLsizei len = 0;
        glGetProgramInfoLog(prog, sizeof(buf), &len, buf);
        gllib::logf(gllib::LogLevel::error, "DepthPyramid %s: %s", name, buf);
        glDeleteProgram(prog);
        return 0;
    }
    return prog;
}

DepthPyramid::DepthPyramid(int width, int height)
    : width_(width)
    , height_(height)
{
    mip_levels_ = 1;
    int w = width, h = height;
    while (w > 1 || h > 1) {
        w = (w > 1) ? (w / 2) : 1;
        h = (h > 1) ? (h / 2) : 1;
        ++mip_levels_;
    }

    glCreateTextures(GL_TEXTURE_2D, 1, &hiz_tex_);
    glTextureStorage2D(hiz_tex_, mip_levels_, GL_R32F, width_, height_);

    glTextureParameteri(hiz_tex_, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
    glTextureParameteri(hiz_tex_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(hiz_tex_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(hiz_tex_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    init_shaders();
}

DepthPyramid::~DepthPyramid() {
    if (hiz_tex_) glDeleteTextures(1, &hiz_tex_);
    if (copy_prog_) glDeleteProgram(copy_prog_);
    if (reduce_prog_) glDeleteProgram(reduce_prog_);
}

DepthPyramid::DepthPyramid(DepthPyramid&& other) noexcept
    : width_(other.width_)
    , height_(other.height_)
    , mip_levels_(other.mip_levels_)
    , hiz_tex_(other.hiz_tex_)
    , copy_prog_(other.copy_prog_)
    , reduce_prog_(other.reduce_prog_)
{
    other.hiz_tex_ = 0;
    other.copy_prog_ = 0;
    other.reduce_prog_ = 0;
}

DepthPyramid& DepthPyramid::operator=(DepthPyramid&& other) noexcept {
    if (this != &other) {
        if (hiz_tex_) glDeleteTextures(1, &hiz_tex_);
        if (copy_prog_) glDeleteProgram(copy_prog_);
        if (reduce_prog_) glDeleteProgram(reduce_prog_);
        width_ = other.width_;
        height_ = other.height_;
        mip_levels_ = other.mip_levels_;
        hiz_tex_ = other.hiz_tex_;
        copy_prog_ = other.copy_prog_;
        reduce_prog_ = other.reduce_prog_;
        other.hiz_tex_ = 0;
        other.copy_prog_ = 0;
        other.reduce_prog_ = 0;
    }
    return *this;
}

void DepthPyramid::init_shaders() {
    copy_prog_ = compile_compute(copy_comp_src, "copy");
    reduce_prog_ = compile_compute(reduce_comp_src, "reduce");
}

void DepthPyramid::build(GLuint depth_texture) {
    if (!copy_prog_ || !reduce_prog_) return;

    glUseProgram(copy_prog_);
    glBindTextureUnit(0, depth_texture);
    glBindImageTexture(1, hiz_tex_, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

    int w = width_, h = height_;
    int groups_x = (w + 15) / 16;
    int groups_y = (h + 15) / 16;
    glDispatchCompute(groups_x, groups_y, 1);

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    int prev_w = w, prev_h = h;
    for (int level = 1; level < mip_levels_; ++level) {
        w = (w > 1) ? (w / 2) : 1;
        h = (h > 1) ? (h / 2) : 1;

        glUseProgram(reduce_prog_);
        glBindImageTexture(0, hiz_tex_, level - 1, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
        glBindImageTexture(1, hiz_tex_, level, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

        groups_x = (w + 15) / 16;
        groups_y = (h + 15) / 16;
        glDispatchCompute(groups_x, groups_y, 1);

        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }
}

void DepthPyramid::bind(int unit) const {
    glBindTextureUnit(unit, hiz_tex_);
}

} // namespace gfx
