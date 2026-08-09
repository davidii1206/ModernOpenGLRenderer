#include "framebuffer.hpp"
#include "texture.hpp"
#include "renderbuffer.hpp"

namespace gl {

Framebuffer::Framebuffer(FramebufferType type)
    : type_(type)
{
    glCreateFramebuffers(1, &handle_);
}

Framebuffer::~Framebuffer() {
    if (handle_) {
        glDeleteFramebuffers(1, &handle_);
    }
}

Framebuffer::Framebuffer(Framebuffer&& other) noexcept
    : handle_(other.handle_)
    , type_(other.type_)
{
    other.handle_ = 0;
}

Framebuffer& Framebuffer::operator=(Framebuffer&& other) noexcept {
    if (this != &other) {
        if (handle_) {
            glDeleteFramebuffers(1, &handle_);
        }
        handle_ = other.handle_;
        type_ = other.type_;
        other.handle_ = 0;
    }
    return *this;
}

void Framebuffer::bind() const {
    glBindFramebuffer(static_cast<GLenum>(type_), handle_);
}

void Framebuffer::unbind(FramebufferType type) {
    glBindFramebuffer(static_cast<GLenum>(type), 0);
}

void Framebuffer::attach_texture(GLenum attachment, const Texture& texture, GLint level) {
    glNamedFramebufferTexture(handle_, attachment, texture.handle(), level);
}

void Framebuffer::attach_texture(GLenum attachment, GLuint texture_handle, GLint level) {
    glNamedFramebufferTexture(handle_, attachment, texture_handle, level);
}

void Framebuffer::attach_renderbuffer(GLenum attachment, const Renderbuffer& rbo) {
    glNamedFramebufferRenderbuffer(handle_, attachment, GL_RENDERBUFFER, rbo.handle());
}

void Framebuffer::draw_buffer(GLenum buf) {
    glNamedFramebufferDrawBuffer(handle_, buf);
}

void Framebuffer::read_buffer(GLenum buf) {
    glNamedFramebufferReadBuffer(handle_, buf);
}

void Framebuffer::no_color_buffer() {
    draw_buffer(GL_NONE);
    read_buffer(GL_NONE);
}

void Framebuffer::blit_to(GLuint dst_fbo, int src_x, int src_y, int src_w, int src_h,
                          int dst_x, int dst_y, int dst_w, int dst_h,
                          GLbitfield mask, GLenum filter) const
{
    glBlitNamedFramebuffer(handle_, dst_fbo,
                           src_x, src_y, src_w, src_h,
                           dst_x, dst_y, dst_w, dst_h,
                           mask, filter);
}

bool Framebuffer::check() const {
    return glCheckNamedFramebufferStatus(handle_, static_cast<GLenum>(type_))
           == GL_FRAMEBUFFER_COMPLETE;
}

} // namespace gl
