#pragma once

#include <glad/glad.h>

namespace gl {

class Renderbuffer {
public:
    Renderbuffer();
    ~Renderbuffer();

    Renderbuffer(const Renderbuffer&) = delete;
    Renderbuffer& operator=(const Renderbuffer&) = delete;

    Renderbuffer(Renderbuffer&& other) noexcept;
    Renderbuffer& operator=(Renderbuffer&& other) noexcept;

    void bind() const;
    static void unbind();

    void storage(GLenum internal_format, GLsizei width, GLsizei height);

    GLuint handle() const { return handle_; }

private:
    GLuint handle_ = 0;
};

} // namespace gl
