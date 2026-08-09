#pragma once

#include <glad/glad.h>
#include <cstddef>
#include <vector>

namespace gl {
class Buffer;
} // namespace gl

namespace gfx {

class Texture;

class BindlessManager {
public:
    explicit BindlessManager(int max_textures = 256);
    ~BindlessManager();

    BindlessManager(const BindlessManager&) = delete;
    BindlessManager& operator=(const BindlessManager&) = delete;

    BindlessManager(BindlessManager&&) noexcept;
    BindlessManager& operator=(BindlessManager&&) noexcept;

    int add(Texture& texture);
    void remove(int index);
    bool occupied(int index) const;
    int count() const { return active_count_; }
    int capacity() const { return max_textures_; }

    void upload();
    void bind(GLuint binding_point = 0) const;

private:
    int max_textures_;
    int active_count_ = 0;
    std::vector<GLuint64> handles_;
    std::vector<bool> occupied_;
    class gl::Buffer* buffer_ = nullptr;
};

} // namespace gfx
