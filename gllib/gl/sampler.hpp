#pragma once

#include <glad/glad.h>

namespace gl {

class Sampler {
public:
    Sampler();
    ~Sampler();

    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

    Sampler(Sampler&& other) noexcept;
    Sampler& operator=(Sampler&& other) noexcept;

    void bind(GLuint unit) const;
    static void unbind(GLuint unit);

    void parameter(GLenum name, GLint value);
    void parameter(GLenum name, GLfloat value);
    void parameter(GLenum name, const GLfloat* values);

    GLuint handle() const { return handle_; }

private:
    GLuint handle_ = 0;
};

} // namespace gl
