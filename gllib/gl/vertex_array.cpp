#include "vertex_array.hpp"

namespace gl {

VertexArray::VertexArray() {
    glCreateVertexArrays(1, &handle_);
}

VertexArray::~VertexArray() {
    if (handle_) {
        glDeleteVertexArrays(1, &handle_);
    }
}

VertexArray::VertexArray(VertexArray&& other) noexcept
    : handle_(other.handle_)
{
    other.handle_ = 0;
}

VertexArray& VertexArray::operator=(VertexArray&& other) noexcept {
    if (this != &other) {
        if (handle_) {
            glDeleteVertexArrays(1, &handle_);
        }
        handle_ = other.handle_;
        other.handle_ = 0;
    }
    return *this;
}

void VertexArray::bind() const {
    glBindVertexArray(handle_);
}

void VertexArray::unbind() {
    glBindVertexArray(0);
}

void VertexArray::attrib_pointer(GLuint index, GLint size, GLenum type,
                                  bool normalized, GLsizei stride, const void* offset)
{
    glVertexAttribPointer(index, size, type, normalized ? GL_TRUE : GL_FALSE, stride, offset);
}

void VertexArray::attrib_i_pointer(GLuint index, GLint size, GLenum type,
                                    GLsizei stride, const void* offset)
{
    glVertexAttribIPointer(index, size, type, stride, offset);
}

void VertexArray::enable_attrib(GLuint index) {
    glEnableVertexAttribArray(index);
}

void VertexArray::disable_attrib(GLuint index) {
    glDisableVertexAttribArray(index);
}

} // namespace gl
