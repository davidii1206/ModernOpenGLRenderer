#include "gpu_particle_system.hpp"
#include "camera.hpp"

#include <gl/buffer.hpp>
#include <gl/program.hpp>
#include <gl/shader.hpp>
#include <gl/state.hpp>
#include <gl/vertex_array.hpp>

#include <cstdio>
#include <cstring>

namespace gfx {

// ── Embedded compute shader: emit + simulate in one dispatch ──
static const char* update_comp_src = R"(
#version 460 core

#define WORKGROUP_SIZE 256

layout(local_size_x = WORKGROUP_SIZE) in;

// Particle layout: positions[i].xyz = pos, .w = 1 (unused)
//                velocities[i].xyz = vel, .w = lifetime remaining
//                colors[i] = rgba, .a = point size
layout(std430, binding = 0) buffer Particles { vec4 positions[]; };
layout(std430, binding = 1) buffer Velocities { vec4 velocities[]; };
layout(std430, binding = 2) buffer Colors { vec4 colors[]; };
layout(std430, binding = 3) buffer Counters {
    // These 4 uints ARE the DrawArraysIndirectCommand
    uint count;              // alive particle count (atomicAdd'd by simulate)
    uint instance_count;     // always 1
    uint first;              // always 0
    uint base_instance;      // always 0
    uint spawn_idx;          // atomic spawn counter
};

uniform float u_dt;
uniform vec3  u_gravity;
uniform float u_min_speed;
uniform float u_max_speed;
uniform float u_spread;
uniform float u_lifetime;
uniform vec3  u_emitter_pos;
uniform vec4  u_color;
uniform uint  u_spawn_count;
uniform uint  u_total_particles;

// Cheap hash for random-like values from a seed
uint hash(uint x) {
    x = (x ^ 61u) ^ (x >> 16u);
    x *= 9u;
    x = x ^ (x >> 4u);
    x *= 0x27d4eb2du;
    x = x ^ (x >> 15u);
    return x;
}
float random(uint seed, float range) {
    return float(hash(seed)) / 4294967296.0 * range;
}
vec3 random_dir(uint seed, float spread) {
    float theta = random(seed, 6.2831853);
    float phi = acos(1.0 - random(seed + 1u, 2.0));
    float x = sin(phi) * cos(theta);
    float y = cos(phi);
    float z = sin(phi) * sin(theta);
    return normalize(vec3(x, y, z)) * spread;
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= u_total_particles) return;

    float dt = u_dt;
    float life = velocities[idx].w;

    // ── Try to spawn a new particle ──
    if (life <= 0.0) {
        uint s = atomicAdd(spawn_idx, 1u);
        if (s < u_spawn_count) {
            // Initialize new particle
            positions[idx] = vec4(u_emitter_pos, 1.0);
            float speed = random(idx + s * 137u, u_max_speed - u_min_speed) + u_min_speed;
            vec3 dir = random_dir(idx + s * 271u, u_spread);
            velocities[idx] = vec4(dir * speed, u_lifetime);
            colors[idx] = u_color;
            atomicAdd(count, 1u);
            return;
        }
        return;
    }

    // ── Simulate alive particle ──
    float new_life = life - dt;
    velocities[idx].w = new_life;
    if (new_life <= 0.0) return; // just died this frame

    vec3 vel = velocities[idx].xyz + u_gravity * dt;
    positions[idx].xyz += vel * dt;
    velocities[idx].xyz = vel;

    float fade = min(new_life / (u_lifetime * 0.2 + 0.001), 1.0);
    colors[idx] = vec4(u_color.rgb * fade, u_color.a);

    atomicAdd(count, 1u);
}
)";

// ── Embedded vertex shader ──
static const char* render_vert_src = R"(
#version 460 core

layout(std430, binding = 0) readonly buffer Positions { vec4 positions[]; };
layout(std430, binding = 1) readonly buffer Velocities { vec4 velocities[]; };
layout(std430, binding = 2) readonly buffer Colors { vec4 colors[]; };

uniform mat4 u_view_proj;

out vec4 v_color;

void main() {
    if (velocities[gl_VertexID].w <= 0.0) {
        gl_Position = vec4(0.0, 0.0, 0.0, -1.0); // clip
        return;
    }
    vec4 pos = positions[gl_VertexID];
    gl_Position = u_view_proj * pos;
    v_color = colors[gl_VertexID];
    gl_PointSize = v_color.a;
}
)";

// ── Embedded fragment shader ──
static const char* render_frag_src = R"(
#version 460 core
in vec4 v_color;
out vec4 frag_color;
void main() {
    float d = length(gl_PointCoord - vec2(0.5));
    if (d > 0.5) discard;
    float a = smoothstep(0.5, 0.0, d);
    frag_color = vec4(v_color.rgb, v_color.a * a);
}
)";

// ── Counter buffer layout ──
// First 4 uints ARE DrawArraysIndirectCommand { count, instance_count, first, base_instance }
// count doubles as alive particle counter (atomically incremented by compute)
struct CounterLayout {
    GLuint count;
    GLuint instance_count;
    GLuint first;
    GLuint base_instance;
    GLuint spawn_idx;
};

// ── Constructor ──
GpuParticleSystem::GpuParticleSystem(int max_particles)
    : max_particles_(max_particles)
{
    init_buffers();
    init_shaders();
}

GpuParticleSystem::~GpuParticleSystem() {
    delete particles_;
    delete counters_;
    delete update_prog_;
    delete render_prog_;
    delete vao_;
}

GpuParticleSystem::GpuParticleSystem(GpuParticleSystem&& other) noexcept
    : max_particles_(other.max_particles_)
    , spawn_accum_(other.spawn_accum_)
    , spawn_rate_(other.spawn_rate_)
    , lifetime_(other.lifetime_)
    , speed_min_(other.speed_min_)
    , speed_max_(other.speed_max_)
    , size_(other.size_)
    , spread_(other.spread_)
    , gravity_(other.gravity_)
    , emitter_pos_(other.emitter_pos_)
    , color_(other.color_)
    , particles_(other.particles_)
    , counters_(other.counters_)
    , update_prog_(other.update_prog_)
    , render_prog_(other.render_prog_)
    , vao_(other.vao_)
    , programs_ok_(other.programs_ok_)
{
    other.particles_ = nullptr;
    other.counters_ = nullptr;
    other.update_prog_ = nullptr;
    other.render_prog_ = nullptr;
    other.vao_ = nullptr;
    other.programs_ok_ = false;
}

GpuParticleSystem& GpuParticleSystem::operator=(GpuParticleSystem&& other) noexcept {
    if (this != &other) {
        delete particles_;
        delete counters_;
        delete update_prog_;
        delete render_prog_;
        delete vao_;

        max_particles_ = other.max_particles_;
        spawn_accum_ = other.spawn_accum_;
        spawn_rate_ = other.spawn_rate_;
        lifetime_ = other.lifetime_;
        speed_min_ = other.speed_min_;
        speed_max_ = other.speed_max_;
        size_ = other.size_;
        spread_ = other.spread_;
        gravity_ = other.gravity_;
        emitter_pos_ = other.emitter_pos_;
        color_ = other.color_;
        particles_ = other.particles_;
        counters_ = other.counters_;
        update_prog_ = other.update_prog_;
        render_prog_ = other.render_prog_;
        vao_ = other.vao_;
        programs_ok_ = other.programs_ok_;

        other.particles_ = nullptr;
        other.counters_ = nullptr;
        other.update_prog_ = nullptr;
        other.render_prog_ = nullptr;
        other.vao_ = nullptr;
        other.programs_ok_ = false;
    }
    return *this;
}

// ── Init ──
void GpuParticleSystem::init_buffers() {
    // Particles: 3x vec4 per particle (pos, vel, color) = 48 bytes
    particles_ = new gl::Buffer(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    particles_->data(nullptr, max_particles_ * 3 * sizeof(glm::vec4));
    // Zero-clear the buffer so dead particles have velocities.w = 0 (life <= 0 == dead)
    static const float zero = 0.0f;
    glClearNamedBufferData(particles_->handle(), GL_R32F, GL_RED, GL_FLOAT, &zero);

    // Counter + indirect draw command
    counters_ = new gl::Buffer(gl::BufferType::shader, gl::BufferUsage::dynamic_draw);
    CounterLayout initial = {};
    initial.instance_count = 1;
    initial.first = 0;
    initial.base_instance = 0;
    counters_->data(&initial, sizeof(CounterLayout));

    vao_ = new gl::VertexArray;
    vao_->bind();
    gl::VertexArray::unbind();
}

void GpuParticleSystem::init_shaders() {
    gl::Shader comp(gl::ShaderType::compute, update_comp_src);
    gl::Shader vert(gl::ShaderType::vertex, render_vert_src);
    gl::Shader frag(gl::ShaderType::fragment, render_frag_src);

    bool ok = true;

    if (comp.compiled()) {
        update_prog_ = new gl::Program;
        update_prog_->attach(comp);
        if (!update_prog_->link()) {
            std::fprintf(stderr, "GpuParticleSystem: update program link failed\n%s\n",
                         update_prog_->info_log().c_str());
            delete update_prog_;
            update_prog_ = nullptr;
            ok = false;
        }
    } else {
        std::fprintf(stderr, "GpuParticleSystem: compute shader compile failed\n");
        ok = false;
    }

    if (vert.compiled() && frag.compiled()) {
        render_prog_ = new gl::Program;
        render_prog_->attach(vert);
        render_prog_->attach(frag);
        if (!render_prog_->link()) {
            std::fprintf(stderr, "GpuParticleSystem: render program link failed\n%s\n",
                         render_prog_->info_log().c_str());
            delete render_prog_;
            render_prog_ = nullptr;
            ok = false;
        }
    } else {
        std::fprintf(stderr, "GpuParticleSystem: render shader compile failed\n");
        ok = false;
    }

    programs_ok_ = ok;
}

// ── Setters ──
void GpuParticleSystem::set_gravity(const glm::vec3& g)          { gravity_ = g; }
void GpuParticleSystem::set_spawn_rate(float rate)              { spawn_rate_ = rate; }
void GpuParticleSystem::set_particle_lifetime(float seconds)    { lifetime_ = seconds; }
void GpuParticleSystem::set_particle_speed(float min, float max){ speed_min_ = min; speed_max_ = max; }
void GpuParticleSystem::set_particle_size(float size)           { size_ = size; }
void GpuParticleSystem::set_color(const glm::vec4& color)       { color_ = color; }
void GpuParticleSystem::set_emitter_position(const glm::vec3& pos){ emitter_pos_ = pos; }
void GpuParticleSystem::set_spread(float spread)                { spread_ = spread; }

// ── Update ──
void GpuParticleSystem::update(float dt) {
    if (!programs_ok_) return;

    // Accumulate spawn count on CPU (simple: how many to emit this frame)
    spawn_accum_ += dt * spawn_rate_;
    int to_spawn = static_cast<int>(spawn_accum_);
    if (to_spawn > 0) {
        spawn_accum_ -= static_cast<float>(to_spawn);
    }
    uint spawn_count = static_cast<uint>(std::max(0, to_spawn));

    // Reset GPU counters for this frame (count=0, spawn_idx=0)
    CounterLayout reset = {};
    reset.instance_count = 1;
    reset.first = 0;
    reset.base_instance = 0;
    reset.spawn_idx = 0;
    counters_->sub_data(&reset, 0, sizeof(CounterLayout));

    // ── Dispatch compute update ──
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    update_prog_->use();

    particles_->bind_base(0);
    // binding 1 = velocities (same buffer, offset by max_particles * vec4)
    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, particles_->handle(),
                      max_particles_ * sizeof(glm::vec4),
                      max_particles_ * sizeof(glm::vec4));
    // binding 2 = colors (offset by max_particles * 2 * vec4)
    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 2, particles_->handle(),
                      max_particles_ * 2 * sizeof(glm::vec4),
                      max_particles_ * sizeof(glm::vec4));
    counters_->bind_base(3);

    auto uniform = [&](const char* name) -> GLint {
        return update_prog_->uniform_location(name);
    };

    GLint loc;
    if ((loc = uniform("u_dt")) >= 0) update_prog_->uniform1f(loc, dt);
    if ((loc = uniform("u_gravity")) >= 0) update_prog_->uniform3f(loc, gravity_.x, gravity_.y, gravity_.z);
    if ((loc = uniform("u_min_speed")) >= 0) update_prog_->uniform1f(loc, speed_min_);
    if ((loc = uniform("u_max_speed")) >= 0) update_prog_->uniform1f(loc, speed_max_);
    if ((loc = uniform("u_spread")) >= 0) update_prog_->uniform1f(loc, spread_);
    if ((loc = uniform("u_lifetime")) >= 0) update_prog_->uniform1f(loc, lifetime_);
    if ((loc = uniform("u_emitter_pos")) >= 0) update_prog_->uniform3f(loc, emitter_pos_.x, emitter_pos_.y, emitter_pos_.z);
    if ((loc = uniform("u_color")) >= 0) {
        glm::vec4 c = color_;
        c.a = size_; // pack point size into alpha
        update_prog_->uniform4f(loc, c.r, c.g, c.b, c.a);
    }
    if ((loc = uniform("u_spawn_count")) >= 0) glProgramUniform1ui(update_prog_->handle(), loc, spawn_count);
    if ((loc = uniform("u_total_particles")) >= 0) glProgramUniform1ui(update_prog_->handle(), loc, static_cast<GLuint>(max_particles_));

    GLuint groups = (static_cast<GLuint>(max_particles_) + 255) / 256;
    gl::dispatch_compute(groups, 1, 1);
    gl::memory_barrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);
}

// ── Render ──
void GpuParticleSystem::render(const Camera& camera) {
    if (!programs_ok_ || !render_prog_) return;

    render_prog_->use();
    particles_->bind_base(0);
    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, particles_->handle(),
                      max_particles_ * sizeof(glm::vec4),
                      max_particles_ * sizeof(glm::vec4));
    glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 2, particles_->handle(),
                      max_particles_ * 2 * sizeof(glm::vec4),
                      max_particles_ * sizeof(glm::vec4));

    GLint loc = render_prog_->uniform_location("u_view_proj");
    if (loc >= 0) render_prog_->uniform_matrix4fv(loc, &camera.view_projection()[0][0]);

    vao_->bind();
    gl::draw_arrays(GL_POINTS, 0, max_particles_);
}

} // namespace gfx
