#pragma once

#include <glad/glad.h>
#include <gl/buffer.hpp>

#include <cstddef>
#include <cstring>

namespace gfx {

template<typename T>
class UniformBuffer {
public:
    UniformBuffer()
        : buffer_(gl::BufferType::uniform, gl::BufferUsage::dynamic_draw)
    {
        buffer_.data(nullptr, sizeof(T));
    }

    void upload(const T& data) {
        buffer_.sub_data(&data, 0, sizeof(T));
    }

    void bind(GLuint binding_point) const {
        buffer_.bind_base(binding_point);
    }

    T* map() {
        return static_cast<T*>(buffer_.map(GL_READ_WRITE));
    }

    void unmap() {
        buffer_.unmap();
    }

    GLuint handle() const { return buffer_.handle(); }

private:
    gl::Buffer buffer_;
};

} // namespace gfx
