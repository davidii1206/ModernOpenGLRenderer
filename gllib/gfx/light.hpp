#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <cstddef>

namespace gl {
class Buffer;
} // namespace gl

namespace gfx {

class LightBuffer {
public:
    static constexpr int MAX_DIRECTIONAL_LIGHTS = 4;
    static constexpr int MAX_POINT_LIGHTS = 16;

    LightBuffer();
    ~LightBuffer() = default;

    LightBuffer(const LightBuffer&) = delete;
    LightBuffer& operator=(const LightBuffer&) = delete;

    LightBuffer(LightBuffer&&) noexcept;
    LightBuffer& operator=(LightBuffer&&) noexcept;

    void set_directional(int index, const glm::vec3& direction,
                         const glm::vec3& color, float intensity = 1.0f);
    void set_point(int index, const glm::vec3& position, const glm::vec3& color,
                   float intensity = 1.0f, float range = 10.0f,
                   float constant_attenuation = 1.0f,
                   float linear_attenuation = 0.09f,
                   float quadratic_attenuation = 0.032f);
    void set_directional_count(int count);
    void set_point_count(int count);

    void upload();
    void bind(GLuint binding_point = 0) const;

    struct alignas(16) DirLight {
        glm::vec4 direction;
        glm::vec4 color;
    };

    struct alignas(16) PointLight {
        glm::vec4 position;
        glm::vec4 color;
        glm::vec4 attenuation;
    };

private:
    DirLight dirs_[MAX_DIRECTIONAL_LIGHTS]{};
    PointLight points_[MAX_POINT_LIGHTS]{};
    int dir_count_ = 0;
    int point_count_ = 0;

    class gl::Buffer* ubo_ = nullptr;
};

} // namespace gfx
