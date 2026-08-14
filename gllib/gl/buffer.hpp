#pragma once

#include <glad/glad.h>
#include <cstddef>

namespace gl {

enum class BufferType : GLenum {
    vertex   = GL_ARRAY_BUFFER,
    index    = GL_ELEMENT_ARRAY_BUFFER,
    uniform  = GL_UNIFORM_BUFFER,
    shader   = GL_SHADER_STORAGE_BUFFER,
    draw     = GL_DRAW_INDIRECT_BUFFER,
    pixel    = GL_PIXEL_UNPACK_BUFFER,
};

enum class BufferUsage : GLenum {
    stream_draw  = GL_STREAM_DRAW,
    stream_read  = GL_STREAM_READ,
    static_draw  = GL_STATIC_DRAW,
    static_read  = GL_STATIC_READ,
    dynamic_draw = GL_DYNAMIC_DRAW,
    dynamic_read = GL_DYNAMIC_READ,
};

class Buffer {
public:
    Buffer();
    Buffer(BufferType type, BufferUsage usage = BufferUsage::static_draw);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    void bind() const;
    static void unbind(BufferType type);

    void bind_base(GLuint index) const;
    void bind_range(GLuint index, GLintptr offset, GLsizeiptr size) const;

    void data(const void* data, std::size_t size);
    void sub_data(const void* data, std::size_t offset, std::size_t size);

    /// Allocate immutable storage with explicit flags (e.g. GL_MAP_PERSISTENT_BIT).
    /// Uses glNamedBufferStorage internally. Overwrites any existing storage.
    void storage(GLsizeiptr size, const void* data, GLbitfield flags);

    void* map(GLenum access);
    void* map_range(GLintptr offset, GLsizeiptr length, GLbitfield access);
    void unmap();

    void clear(GLenum internal_format, GLenum format, GLenum type,
               const void* data);

    GLuint handle() const { return handle_; }
    std::size_t size() const { return size_; }
    BufferType type() const { return type_; }

private:
    GLuint handle_ = 0;
    BufferType type_;
    BufferUsage usage_;
    std::size_t size_ = 0;
};

} // namespace gl
