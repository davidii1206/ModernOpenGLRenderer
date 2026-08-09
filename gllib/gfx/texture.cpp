#include "texture.hpp"

#include <gllib/log.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace gfx {

Texture::~Texture() {
    if (bindless_handle_ && resident_) {
        glMakeTextureHandleNonResidentARB(bindless_handle_);
    }
    if (handle_) {
        glDeleteTextures(1, &handle_);
    }
}

Texture::Texture(Texture&& other) noexcept
    : handle_(other.handle_)
    , width_(other.width_)
    , height_(other.height_)
    , channels_(other.channels_)
    , bindless_handle_(other.bindless_handle_)
    , resident_(other.resident_)
{
    other.handle_ = 0;
    other.width_ = 0;
    other.height_ = 0;
    other.channels_ = 0;
    other.bindless_handle_ = 0;
    other.resident_ = false;
}

Texture& Texture::operator=(Texture&& other) noexcept {
    if (this != &other) {
        if (bindless_handle_ && resident_) {
            glMakeTextureHandleNonResidentARB(bindless_handle_);
        }
        if (handle_) {
            glDeleteTextures(1, &handle_);
        }
        handle_ = other.handle_;
        width_ = other.width_;
        height_ = other.height_;
        channels_ = other.channels_;
        bindless_handle_ = other.bindless_handle_;
        resident_ = other.resident_;
        other.handle_ = 0;
        other.width_ = 0;
        other.height_ = 0;
        other.channels_ = 0;
        other.bindless_handle_ = 0;
        other.resident_ = false;
    }
    return *this;
}

bool Texture::load(std::string_view path) {
    int w, h, ch;
    unsigned char* data = stbi_load(path.data(), &w, &h, &ch, 4);
    if (!data) {
        gllib::logf(gllib::LogLevel::error, "failed to load texture: %s", path.data());
        return false;
    }

    width_ = w;
    height_ = h;
    channels_ = 4;

    glCreateTextures(GL_TEXTURE_2D, 1, &handle_);
    glTextureStorage2D(handle_, 1, GL_SRGB8_ALPHA8, width_, height_);
    glTextureSubImage2D(handle_, 0, 0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, data);

    glTextureParameteri(handle_, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTextureParameteri(handle_, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(handle_, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTextureParameteri(handle_, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glGenerateTextureMipmap(handle_);

    stbi_image_free(data);
    return true;
}

void Texture::create(int width, int height, GLenum internal_format) {
    width_ = width;
    height_ = height;
    channels_ = 0;

    glCreateTextures(GL_TEXTURE_2D, 1, &handle_);
    glTextureStorage2D(handle_, 1, internal_format, width_, height_);
}

void Texture::upload(const void* data, GLenum format, GLenum type, int level) {
    glTextureSubImage2D(handle_, level, 0, 0, width_, height_, format, type, data);
}

void Texture::parameter(GLenum name, GLint value) {
    glTextureParameteri(handle_, name, value);
}

void Texture::parameter(GLenum name, GLfloat value) {
    glTextureParameterf(handle_, name, value);
}

void Texture::generate_mipmap() {
    glGenerateTextureMipmap(handle_);
}

void Texture::bind(int unit) const {
    glBindTextureUnit(static_cast<GLuint>(unit), handle_);
}

GLuint64 Texture::bindless_handle() {
    if (!bindless_handle_) {
        bindless_handle_ = glGetTextureHandleARB(handle_);
    }
    return bindless_handle_;
}

void Texture::make_resident() {
    if (!resident_) {
        glMakeTextureHandleResidentARB(bindless_handle());
        resident_ = true;
    }
}

void Texture::make_non_resident() {
    if (resident_) {
        glMakeTextureHandleNonResidentARB(bindless_handle_);
        resident_ = false;
    }
}

} // namespace gfx
