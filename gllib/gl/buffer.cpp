#include "buffer.hpp"

namespace gl {

Buffer::Buffer()
    : Buffer(BufferType::shader, BufferUsage::dynamic_draw)
{
}

Buffer::Buffer(BufferType type, BufferUsage usage)
    : type_(type), usage_(usage)
{
    glCreateBuffers(1, &handle_);
}

Buffer::~Buffer() {
    if (handle_) {
        glDeleteBuffers(1, &handle_);
    }
}

Buffer::Buffer(Buffer&& other) noexcept
    : handle_(other.handle_)
    , type_(other.type_)
    , usage_(other.usage_)
    , size_(other.size_)
{
    other.handle_ = 0;
    other.size_ = 0;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        if (handle_) {
            glDeleteBuffers(1, &handle_);
        }
        handle_ = other.handle_;
        type_ = other.type_;
        usage_ = other.usage_;
        size_ = other.size_;
        other.handle_ = 0;
        other.size_ = 0;
    }
    return *this;
}

void Buffer::bind() const {
    glBindBuffer(static_cast<GLenum>(type_), handle_);
}

void Buffer::unbind(BufferType type) {
    glBindBuffer(static_cast<GLenum>(type), 0);
}

void Buffer::bind_base(GLuint index) const {
    glBindBufferBase(static_cast<GLenum>(type_), index, handle_);
}

void Buffer::bind_range(GLuint index, GLintptr offset, GLsizeiptr size) const {
    glBindBufferRange(static_cast<GLenum>(type_), index, handle_, offset, size);
}

void Buffer::data(const void* data, std::size_t size) {
    size_ = size;
    glNamedBufferData(handle_, size, data, static_cast<GLenum>(usage_));
}

void Buffer::sub_data(const void* data, std::size_t offset, std::size_t size) {
    glNamedBufferSubData(handle_, offset, size, data);
}

void Buffer::storage(GLsizeiptr size, const void* data, GLbitfield flags) {
    size_ = size;
    glNamedBufferStorage(handle_, size, data, flags);
}

void* Buffer::map(GLenum access) {
    return glMapNamedBuffer(handle_, access);
}

void* Buffer::map_range(GLintptr offset, GLsizeiptr length, GLbitfield access) {
    return glMapNamedBufferRange(handle_, offset, length, access);
}

void Buffer::unmap() {
    glUnmapNamedBuffer(handle_);
}

void Buffer::clear(GLenum internal_format, GLenum format, GLenum type,
                   const void* data)
{
    glClearNamedBufferData(handle_, internal_format, format, type, data);
}

} // namespace gl
