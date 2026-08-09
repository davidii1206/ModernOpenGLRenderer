#include "renderbuffer.hpp"

namespace gl {

Renderbuffer::Renderbuffer() {
    glCreateRenderbuffers(1, &handle_);
}

Renderbuffer::~Renderbuffer() {
    if (handle_) {
        glDeleteRenderbuffers(1, &handle_);
    }
}

Renderbuffer::Renderbuffer(Renderbuffer&& other) noexcept
    : handle_(other.handle_)
{
    other.handle_ = 0;
}

Renderbuffer& Renderbuffer::operator=(Renderbuffer&& other) noexcept {
    if (this != &other) {
        if (handle_) {
            glDeleteRenderbuffers(1, &handle_);
        }
        handle_ = other.handle_;
        other.handle_ = 0;
    }
    return *this;
}

void Renderbuffer::bind() const {
    glBindRenderbuffer(GL_RENDERBUFFER, handle_);
}

void Renderbuffer::unbind() {
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
}

void Renderbuffer::storage(GLenum internal_format, GLsizei width, GLsizei height) {
    glNamedRenderbufferStorage(handle_, internal_format, width, height);
}

} // namespace gl
