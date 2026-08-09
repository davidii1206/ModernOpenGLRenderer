#include "bindless.hpp"
#include "texture.hpp"

#include <gl/buffer.hpp>

#include <algorithm>

namespace gfx {

BindlessManager::BindlessManager(int max_textures)
    : max_textures_(max_textures)
    , handles_(max_textures, 0)
    , occupied_(max_textures, false)
    , buffer_(new gl::Buffer(gl::BufferType::shader, gl::BufferUsage::dynamic_draw))
{
    buffer_->data(nullptr, max_textures * sizeof(GLuint64));
}

BindlessManager::~BindlessManager() {
    delete buffer_;
}

BindlessManager::BindlessManager(BindlessManager&& other) noexcept
    : max_textures_(other.max_textures_)
    , active_count_(other.active_count_)
    , handles_(std::move(other.handles_))
    , occupied_(std::move(other.occupied_))
    , buffer_(other.buffer_)
{
    other.buffer_ = nullptr;
    other.max_textures_ = 0;
    other.active_count_ = 0;
}

BindlessManager& BindlessManager::operator=(BindlessManager&& other) noexcept {
    if (this != &other) {
        delete buffer_;

        max_textures_ = other.max_textures_;
        active_count_ = other.active_count_;
        handles_ = std::move(other.handles_);
        occupied_ = std::move(other.occupied_);
        buffer_ = other.buffer_;

        other.buffer_ = nullptr;
        other.max_textures_ = 0;
        other.active_count_ = 0;
    }
    return *this;
}

int BindlessManager::add(Texture& texture) {
    auto it = std::find(occupied_.begin(), occupied_.end(), false);
    if (it == occupied_.end()) return -1;

    int index = static_cast<int>(std::distance(occupied_.begin(), it));
    handles_[index] = texture.bindless_handle();
    texture.make_resident();
    occupied_[index] = true;
    ++active_count_;
    return index;
}

void BindlessManager::remove(int index) {
    if (index < 0 || index >= max_textures_ || !occupied_[index]) return;
    if (handles_[index] != 0) {
        glMakeTextureHandleNonResidentARB(handles_[index]);
    }
    handles_[index] = 0;
    occupied_[index] = false;
    --active_count_;
}

bool BindlessManager::occupied(int index) const {
    if (index < 0 || index >= max_textures_) return false;
    return occupied_[index];
}

void BindlessManager::upload() {
    buffer_->sub_data(handles_.data(), 0, max_textures_ * sizeof(GLuint64));
}

void BindlessManager::bind(GLuint binding_point) const {
    buffer_->bind_base(binding_point);
}

} // namespace gfx
