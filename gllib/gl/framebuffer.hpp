#pragma once

#include <glad/glad.h>

namespace gl {

class Texture;
class Renderbuffer;

enum class FramebufferType : GLenum {
    draw      = GL_DRAW_FRAMEBUFFER,
    read      = GL_READ_FRAMEBUFFER,
    both      = GL_FRAMEBUFFER,
};

class Framebuffer {
public:
    explicit Framebuffer(FramebufferType type = FramebufferType::both);
    ~Framebuffer();

    Framebuffer(const Framebuffer&) = delete;
    Framebuffer& operator=(const Framebuffer&) = delete;

    Framebuffer(Framebuffer&& other) noexcept;
    Framebuffer& operator=(Framebuffer&& other) noexcept;

    void bind() const;
    static void unbind(FramebufferType type);

    void attach_texture(GLenum attachment, const Texture& texture, GLint level = 0);
    void attach_texture(GLenum attachment, GLuint texture_handle, GLint level = 0);
    void attach_renderbuffer(GLenum attachment, const class Renderbuffer& rbo);

    void draw_buffer(GLenum buf);
    void read_buffer(GLenum buf);
    void no_color_buffer();

    void blit_to(GLuint dst_fbo, int src_x, int src_y, int src_w, int src_h,
                 int dst_x, int dst_y, int dst_w, int dst_h,
                 GLbitfield mask, GLenum filter) const;

    bool check() const;

    GLuint handle() const { return handle_; }
    FramebufferType type() const { return type_; }

private:
    GLuint handle_ = 0;
    FramebufferType type_;
};

} // namespace gl
