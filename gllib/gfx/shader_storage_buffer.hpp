#pragma once

#include <glad/glad.h>
#include <gl/buffer.hpp>

#include <cstddef>
#include <vector>

namespace gfx {

template<typename T>
class ShaderStorageBuffer {
public:
    ShaderStorageBuffer(int reserve = 0)
        : buffer_(gl::BufferType::shader, gl::BufferUsage::dynamic_draw)
    {
        if (reserve > 0) {
            data_.reserve(reserve);
        }
    }

    void push_back(const T& item) {
        data_.push_back(item);
        dirty_ = true;
    }

    void clear() {
        data_.clear();
        dirty_ = true;
    }

    void resize(std::size_t count) {
        data_.resize(count);
        dirty_ = true;
    }

    T& operator[](std::size_t index) { return data_[index]; }
    const T& operator[](std::size_t index) const { return data_[index]; }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    std::size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }

    void upload() {
        if (data_.empty()) {
            buffer_.data(nullptr, 0);
        } else {
            buffer_.data(data_.data(), data_.size() * sizeof(T));
        }
        dirty_ = false;
    }

    void bind(GLuint binding_point) {
        if (dirty_) upload();
        buffer_.bind_base(binding_point);
    }

    GLuint handle() const { return buffer_.handle(); }

private:
    std::vector<T> data_;
    gl::Buffer buffer_;
    bool dirty_ = true;
};

} // namespace gfx
