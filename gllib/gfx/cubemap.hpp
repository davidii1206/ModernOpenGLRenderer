#pragma once

#include <glad/glad.h>
#include <string>

namespace gfx {

class Cubemap {
public:
    Cubemap() = default;
    ~Cubemap();

    Cubemap(const Cubemap&) = delete;
    Cubemap& operator=(const Cubemap&) = delete;

    Cubemap(Cubemap&& other) noexcept;
    Cubemap& operator=(Cubemap&& other) noexcept;

    bool load_from_files(const std::string& pos_x, const std::string& neg_x,
                         const std::string& pos_y, const std::string& neg_y,
                         const std::string& pos_z, const std::string& neg_z);

    void generate_procedural(int size = 256);
    void generate_procedural_hdr(int size = 256, int levels = 1);

    void create_storage(int size, int levels, GLenum internal_format = GL_RGBA16F);
    void bind_image(int face, int level, int unit, GLenum access, GLenum format) const;
    void bind_image_layered(int level, int unit, GLenum access, GLenum format) const;

    void bind(int unit = 0) const;
    void parameter(GLenum name, GLint value);
    void generate_mipmap();

    GLuint handle() const { return handle_; }
    int width() const { return width_; }
    int height() const { return height_; }
    int levels() const { return levels_; }

private:
    void upload_face(GLenum target, int w, int h, const unsigned char* data);
    void upload_face_float(GLenum target, int w, int h, const float* data);

    GLuint handle_ = 0;
    int width_ = 0;
    int height_ = 0;
    int levels_ = 1;
};

} // namespace gfx
