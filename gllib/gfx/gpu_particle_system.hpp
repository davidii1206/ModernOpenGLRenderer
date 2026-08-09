#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <memory>

namespace gl {
class Buffer;
class Program;
class VertexArray;
} // namespace gl

namespace gfx {

class Camera;

class GpuParticleSystem {
public:
    explicit GpuParticleSystem(int max_particles = 65536);
    ~GpuParticleSystem();

    GpuParticleSystem(const GpuParticleSystem&) = delete;
    GpuParticleSystem& operator=(const GpuParticleSystem&) = delete;

    GpuParticleSystem(GpuParticleSystem&&) noexcept;
    GpuParticleSystem& operator=(GpuParticleSystem&&) noexcept;

    // Configuration
    void set_gravity(const glm::vec3& g);
    void set_spawn_rate(float particles_per_second);
    void set_particle_lifetime(float seconds);
    void set_particle_speed(float min, float max);
    void set_particle_size(float size);
    void set_color(const glm::vec4& color);
    void set_emitter_position(const glm::vec3& pos);
    void set_spread(float spread);

    // Core API
    void update(float dt);
    void render(const Camera& camera);

private:
    void init_buffers();
    void init_shaders();
    void ensure_programs();

    int max_particles_;
    float spawn_accum_ = 0.0f;
    float spawn_rate_  = 0.0f;
    float lifetime_    = 2.0f;
    float speed_min_   = 1.0f;
    float speed_max_   = 3.0f;
    float size_        = 8.0f;
    float spread_      = 1.0f;
    glm::vec3 gravity_{0.0f, -9.8f, 0.0f};
    glm::vec3 emitter_pos_{0.0f, 0.0f, 0.0f};
    glm::vec4 color_{1.0f, 1.0f, 1.0f, 1.0f};

    gl::Buffer* particles_    = nullptr;  // vec4 pos, vec4 vel, vec4 color per particle
    gl::Buffer* counters_     = nullptr;  // alive_count (uint), spawn_idx (uint), indirect cmd
    gl::Program* update_prog_ = nullptr;
    gl::Program* render_prog_ = nullptr;
    gl::VertexArray* vao_     = nullptr;
    bool programs_ok_ = false;
};

} // namespace gfx
