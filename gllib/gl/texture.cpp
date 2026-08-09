#include "texture.hpp"

namespace gl {

Texture::Texture(TextureType type)
    : type_(type)
{
    glCreateTextures(static_cast<GLenum>(type_), 1, &handle_);
}

Texture::~Texture() {
    if (handle_) {
        glDeleteTextures(1, &handle_);
    }
}

Texture::Texture(Texture&& other) noexcept
    : handle_(other.handle_)
    , type_(other.type_)
{
    other.handle_ = 0;
}

Texture& Texture::operator=(Texture&& other) noexcept {
    if (this != &other) {
        if (handle_) {
            glDeleteTextures(1, &handle_);
        }
        handle_ = other.handle_;
        type_ = other.type_;
        other.handle_ = 0;
    }
    return *this;
}

void Texture::bind() const {
    glBindTextureUnit(0, handle_);
}

void Texture::bind(int unit) const {
    glBindTextureUnit(static_cast<GLuint>(unit), handle_);
}

void Texture::unbind(TextureType) {
    glBindTextureUnit(0, 0);
}

void Texture::image_2d(GLint level, GLint internal_format,
                        GLsizei width, GLsizei height,
                        GLenum format, GLenum data_type,
                        const void* data,
                        GLsizei mip_levels)
{
    glTextureStorage2D(handle_, mip_levels, internal_format, width, height);
    if (data) {
        glTextureSubImage2D(handle_, level, 0, 0, width, height,
                            format, data_type, data);
    }
}

void Texture::bind_image(GLuint unit, GLint level, GLboolean layered, GLint layer,
                          GLenum access, GLenum format) const
{
    glBindImageTexture(unit, handle_, level, layered, layer, access, format);
}

void Texture::sub_image_2d(GLint level, GLint xoffset, GLint yoffset,
                            GLsizei width, GLsizei height,
                            GLenum format, GLenum data_type,
                            const void* data)
{
    glTextureSubImage2D(handle_, level, xoffset, yoffset,
                        width, height, format, data_type, data);
}

void Texture::parameter(GLenum name, GLint value) {
    glTextureParameteri(handle_, name, value);
}

void Texture::parameter(GLenum name, GLfloat value) {
    glTextureParameterf(handle_, name, value);
}

void Texture::parameter(GLenum name, const GLfloat* values) {
    glTextureParameterfv(handle_, name, values);
}

void Texture::generate_mipmap() {
    glGenerateTextureMipmap(handle_);
}

} // namespace gl
