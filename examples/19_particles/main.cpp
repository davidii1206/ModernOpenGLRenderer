// Example 19 — gfx::GpuParticleSystem: fully GPU-driven particles.

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>

#include <glm/gtc/matrix_transform.hpp>

#include <cstdlib>

int main() {
    gfx::Window window({"19 Particles", 800, 600});

    gfx::GpuParticleSystem ps(65536);
    ps.set_gravity({0.0f, -5.0f, 0.0f});
    ps.set_spawn_rate(800.0f);
    ps.set_particle_lifetime(4.0f);
    ps.set_particle_speed(1.0f, 3.0f);
    ps.set_particle_size(12.0f);
    ps.set_color({0.4f, 0.7f, 1.0f, 1.0f});
    ps.set_spread(0.8f);
    ps.set_emitter_position({0.0f, 2.0f, 0.0f});

    gfx::Camera cam;
    cam.perspective(45.0f, float(window.width()) / window.height(), 0.1f, 100.0f);
    cam.look_at({4, 3, 6}, {0, 1, 0});

    gfx::Renderer renderer;
    renderer.set_clear_color(0.02f, 0.02f, 0.04f, 1.0f);
    gl::enable(GL_PROGRAM_POINT_SIZE);
    gl::enable(GL_BLEND);
    gl::blend_func(GL_SRC_ALPHA, GL_ONE);
    gl::depth_mask(GL_FALSE);

    float last = window.time();
    while (!window.should_close()) {
        float now = window.time();
        float dt = now - last;
        last = now;

        window.poll_events();
        cam.set_aspect(float(window.width()) / window.height());

        renderer.clear(GL_COLOR_BUFFER_BIT);
        ps.update(dt);
        ps.render(cam);

        window.swap_buffers();
    }

    return EXIT_SUCCESS;
}
