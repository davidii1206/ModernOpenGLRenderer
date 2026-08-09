#include "gbuffer.hpp"
#include "../gl/state.hpp"
#include <cstdio>

namespace gfx {

GBuffer::~GBuffer() {
    if (fbo_) glDeleteFramebuffers(1, &fbo_);
    if (albedo_) glDeleteTextures(1, &albedo_);
    if (normal_rm_) glDeleteTextures(1, &normal_rm_);
    if (emissive_ao_) glDeleteTextures(1, &emissive_ao_);
    if (depth_) glDeleteTextures(1, &depth_);
}

GBuffer::GBuffer(GBuffer&& other) noexcept
    : fbo_(other.fbo_), albedo_(other.albedo_), normal_rm_(other.normal_rm_),
      emissive_ao_(other.emissive_ao_), depth_(other.depth_),
      width_(other.width_), height_(other.height_)
{
    other.fbo_ = other.albedo_ = other.normal_rm_ = other.emissive_ao_ = other.depth_ = 0;
    other.width_ = other.height_ = 0;
}

GBuffer& GBuffer::operator=(GBuffer&& other) noexcept {
    if (this != &other) {
        if (fbo_) glDeleteFramebuffers(1, &fbo_);
        if (albedo_) glDeleteTextures(1, &albedo_);
        if (normal_rm_) glDeleteTextures(1, &normal_rm_);
        if (emissive_ao_) glDeleteTextures(1, &emissive_ao_);
        if (depth_) glDeleteTextures(1, &depth_);
        fbo_ = other.fbo_; albedo_ = other.albedo_; normal_rm_ = other.normal_rm_;
        emissive_ao_ = other.emissive_ao_; depth_ = other.depth_;
        width_ = other.width_; height_ = other.height_;
        other.fbo_ = other.albedo_ = other.normal_rm_ = other.emissive_ao_ = other.depth_ = 0;
        other.width_ = other.height_ = 0;
    }
    return *this;
}

void GBuffer::create(int width, int height) {
    width_ = width;
    height_ = height;

    // Albedo — RT0
    glCreateTextures(GL_TEXTURE_2D, 1, &albedo_);
    glTextureStorage2D(albedo_, 1, GL_SRGB8_ALPHA8, width, height);
    glTextureParameteri(albedo_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(albedo_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(albedo_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(albedo_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Normal.xy + Roughness + Metallic — RT1
    glCreateTextures(GL_TEXTURE_2D, 1, &normal_rm_);
    glTextureStorage2D(normal_rm_, 1, GL_RGBA16F, width, height);
    glTextureParameteri(normal_rm_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(normal_rm_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(normal_rm_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(normal_rm_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Emissive.rgb + AO — RT2
    glCreateTextures(GL_TEXTURE_2D, 1, &emissive_ao_);
    glTextureStorage2D(emissive_ao_, 1, GL_RGBA16F, width, height);
    glTextureParameteri(emissive_ao_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(emissive_ao_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(emissive_ao_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(emissive_ao_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Depth
    glCreateTextures(GL_TEXTURE_2D, 1, &depth_);
    glTextureStorage2D(depth_, 1, GL_DEPTH_COMPONENT24, width, height);
    glTextureParameteri(depth_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(depth_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(depth_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(depth_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // FBO
    glCreateFramebuffers(1, &fbo_);
    glNamedFramebufferTexture(fbo_, GL_COLOR_ATTACHMENT0, albedo_, 0);
    glNamedFramebufferTexture(fbo_, GL_COLOR_ATTACHMENT1, normal_rm_, 0);
    glNamedFramebufferTexture(fbo_, GL_COLOR_ATTACHMENT2, emissive_ao_, 0);
    glNamedFramebufferTexture(fbo_, GL_DEPTH_ATTACHMENT, depth_, 0);

    GLenum bufs[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
    glNamedFramebufferDrawBuffers(fbo_, 3, bufs);
    glNamedFramebufferReadBuffer(fbo_, GL_NONE);

    GLenum status = glCheckNamedFramebufferStatus(fbo_, GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        std::fprintf(stderr, "GBuffer FBO incomplete: 0x%x\n", status);
    }
}

void GBuffer::bind_for_geometry() {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glViewport(0, 0, width_, height_);
    gl::enable(GL_DEPTH_TEST);
    gl::depth_func(GL_LESS);
    gl::depth_mask(GL_TRUE);
    gl::disable(GL_CULL_FACE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void GBuffer::bind_for_lighting(int albedo_unit, int normal_rm_unit,
                                int emissive_ao_unit, int depth_unit) {
    glBindTextureUnit(albedo_unit, albedo_);
    glBindTextureUnit(normal_rm_unit, normal_rm_);
    glBindTextureUnit(emissive_ao_unit, emissive_ao_);
    glBindTextureUnit(depth_unit, depth_);
}

} // namespace gfx
