#include "light.hpp"

#include <glad/glad.h>
#include <gl/buffer.hpp>

#include <cstring>

namespace gfx {

static constexpr std::size_t total_ubo_size =
    sizeof(LightBuffer::DirLight) * LightBuffer::MAX_DIRECTIONAL_LIGHTS +
    sizeof(LightBuffer::PointLight) * LightBuffer::MAX_POINT_LIGHTS +
    4 * sizeof(int);

LightBuffer::LightBuffer()
    : ubo_(new gl::Buffer(gl::BufferType::uniform, gl::BufferUsage::dynamic_draw))
{
    ubo_->data(nullptr, total_ubo_size);
}

LightBuffer::LightBuffer(LightBuffer&& other) noexcept
    : ubo_(other.ubo_)
    , dir_count_(other.dir_count_)
    , point_count_(other.point_count_)
{
    std::memcpy(dirs_, other.dirs_, sizeof(dirs_));
    std::memcpy(points_, other.points_, sizeof(points_));
    other.ubo_ = nullptr;
}

LightBuffer& LightBuffer::operator=(LightBuffer&& other) noexcept {
    if (this != &other) {
        delete ubo_;
        ubo_ = other.ubo_;
        other.ubo_ = nullptr;
        dir_count_ = other.dir_count_;
        point_count_ = other.point_count_;
        std::memcpy(dirs_, other.dirs_, sizeof(dirs_));
        std::memcpy(points_, other.points_, sizeof(points_));
    }
    return *this;
}

void LightBuffer::set_directional(int index, const glm::vec3& dir,
                                   const glm::vec3& color, float intensity) {
    if (index < 0 || index >= MAX_DIRECTIONAL_LIGHTS) return;
    dirs_[index].direction = glm::vec4(dir, 0.0f);
    dirs_[index].color = glm::vec4(color, intensity);
}

void LightBuffer::set_point(int index, const glm::vec3& position,
                             const glm::vec3& color, float intensity,
                             float range, float constant_attenuation,
                             float linear_attenuation,
                             float quadratic_attenuation) {
    if (index < 0 || index >= MAX_POINT_LIGHTS) return;
    points_[index].position = glm::vec4(position, 1.0f);
    points_[index].color = glm::vec4(color, intensity);
    points_[index].attenuation = glm::vec4(constant_attenuation, linear_attenuation,
                                            quadratic_attenuation, range);
}

void LightBuffer::set_directional_count(int count) {
    dir_count_ = count;
}

void LightBuffer::set_point_count(int count) {
    point_count_ = count;
}

void LightBuffer::upload() {
    std::size_t offset = 0;
    ubo_->sub_data(dirs_, offset, sizeof(dirs_));
    offset += sizeof(dirs_);
    ubo_->sub_data(points_, offset, sizeof(points_));
    offset += sizeof(points_);

    int counts[4] = { dir_count_, point_count_, 0, 0 };
    ubo_->sub_data(counts, offset, sizeof(counts));
}

void LightBuffer::bind(GLuint binding_point) const {
    ubo_->bind_base(binding_point);
}

} // namespace gfx
