#pragma once

#include <glad/glad.h>
#include <string_view>

namespace gfx {

struct TextureDesc {
    int width = 0;
    int height = 0;
    int channels = 0;
    GLenum internal_format = GL_SRGB8_ALPHA8;
    bool generate_mipmaps = true;
    bool bindless = false;
};

class Texture {
public:
    Texture() = default;
    ~Texture();

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    Texture(Texture&& other) noexcept;
    Texture& operator=(Texture&& other) noexcept;

    bool load(std::string_view path);
    void create(int width, int height, GLenum internal_format);
    void upload(const void* data, GLenum format = GL_RGBA, GLenum type = GL_UNSIGNED_BYTE,
                int level = 0);

    void parameter(GLenum name, GLint value);
    void parameter(GLenum name, GLfloat value);
    void generate_mipmap();

    void bind(int unit = 0) const;

    GLuint64 bindless_handle();
    void make_resident();
    void make_non_resident();

    GLuint handle() const { return handle_; }
    int width() const { return width_; }
    int height() const { return height_; }

private:
    GLuint handle_ = 0;
    int width_ = 0;
    int height_ = 0;
    int channels_ = 0;
    GLuint64 bindless_handle_ = 0;
    bool resident_ = false;
};

} // namespace gfx
