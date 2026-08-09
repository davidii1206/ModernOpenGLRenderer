#include "sampler.hpp"

namespace gl {

Sampler::Sampler() {
    glCreateSamplers(1, &handle_);
}

Sampler::~Sampler() {
    if (handle_) {
        glDeleteSamplers(1, &handle_);
    }
}

Sampler::Sampler(Sampler&& other) noexcept
    : handle_(other.handle_)
{
    other.handle_ = 0;
}

Sampler& Sampler::operator=(Sampler&& other) noexcept {
    if (this != &other) {
        if (handle_) {
            glDeleteSamplers(1, &handle_);
        }
        handle_ = other.handle_;
        other.handle_ = 0;
    }
    return *this;
}

void Sampler::bind(GLuint unit) const {
    glBindSampler(unit, handle_);
}

void Sampler::unbind(GLuint unit) {
    glBindSampler(unit, 0);
}

void Sampler::parameter(GLenum name, GLint value) {
    glSamplerParameteri(handle_, name, value);
}

void Sampler::parameter(GLenum name, GLfloat value) {
    glSamplerParameterf(handle_, name, value);
}

void Sampler::parameter(GLenum name, const GLfloat* values) {
    glSamplerParameterfv(handle_, name, values);
}

} // namespace gl
