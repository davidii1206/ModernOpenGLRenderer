#pragma once

#include <glad/glad.h>
#include <cstddef>

namespace gl {

class VertexArray {
public:
    VertexArray();
    ~VertexArray();

    VertexArray(const VertexArray&) = delete;
    VertexArray& operator=(const VertexArray&) = delete;

    VertexArray(VertexArray&& other) noexcept;
    VertexArray& operator=(VertexArray&& other) noexcept;

    void bind() const;
    static void unbind();

    void attrib_pointer(GLuint index, GLint size, GLenum type,
                        bool normalized, GLsizei stride, const void* offset);
    void attrib_i_pointer(GLuint index, GLint size, GLenum type,
                          GLsizei stride, const void* offset);
    void enable_attrib(GLuint index);
    void disable_attrib(GLuint index);

    GLuint handle() const { return handle_; }

private:
    GLuint handle_ = 0;
};

} // namespace gl
