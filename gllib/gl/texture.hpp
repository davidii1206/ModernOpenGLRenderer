#pragma once

#include <glad/glad.h>
#include <cstddef>

namespace gl {

enum class TextureType : GLenum {
    tex_1d      = GL_TEXTURE_1D,
    tex_2d      = GL_TEXTURE_2D,
    tex_3d      = GL_TEXTURE_3D,
    cube_map    = GL_TEXTURE_CUBE_MAP,
    tex_1d_arr  = GL_TEXTURE_1D_ARRAY,
    tex_2d_arr  = GL_TEXTURE_2D_ARRAY,
    tex_2d_ms   = GL_TEXTURE_2D_MULTISAMPLE,
};

class Texture {
public:
    explicit Texture(TextureType type = TextureType::tex_2d);
    ~Texture();

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    Texture(Texture&& other) noexcept;
    Texture& operator=(Texture&& other) noexcept;

    void bind() const;
    void bind(int unit) const;
    static void unbind(TextureType type);

    void image_2d(GLint level, GLint internal_format,
                  GLsizei width, GLsizei height,
                  GLenum format, GLenum data_type,
                  const void* data,
                  GLsizei mip_levels = 1);

    void sub_image_2d(GLint level, GLint xoffset, GLint yoffset,
                      GLsizei width, GLsizei height,
                      GLenum format, GLenum data_type,
                      const void* data);

    void bind_image(GLuint unit, GLint level, GLboolean layered, GLint layer,
                    GLenum access, GLenum format) const;

    void parameter(GLenum name, GLint value);
    void parameter(GLenum name, GLfloat value);
    void parameter(GLenum name, const GLfloat* values);

    void generate_mipmap();

    GLuint handle() const { return handle_; }
    TextureType type() const { return type_; }

private:
    GLuint handle_ = 0;
    TextureType type_;
};

} // namespace gl
