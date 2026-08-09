// Example 24 — gfx::DebugDraw: immediate-mode line, box, sphere, axis, frustum.

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>

#include <cstdlib>
#include <cstdio>

int main() {
    gfx::WindowDesc wdesc;
    wdesc.title = "24 Debug Draw";
    wdesc.width = 1024;
    wdesc.height = 768;
    wdesc.debug = true;
    gfx::Window window(wdesc);

    gl::enable_debug_output();

    gfx::Camera cam;
    cam.perspective(45.0f, float(window.width()) / window.height(), 0.1f, 100.0f);
    cam.look_at({8, 6, 10}, {0, 0, 0});

    gfx::Renderer renderer;
    renderer.set_clear_color(0.05f, 0.05f, 0.1f, 1.0f);

    gfx::DebugDraw dd;

    float elapsed = 0.0f;
    while (!window.should_close()) {
        float dt = window.time();
        window.poll_events();
        cam.set_aspect(float(window.width()) / window.height());

        elapsed += dt;

        renderer.clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // ── Debug primitives ──
        dd.clear();

        // World axis
        dd.draw_axis(glm::mat4(1.0f), 8.0f);

        // Box at origin
        dd.draw_box({-2, -1, -2}, {2, 1, 2}, {1, 1, 0, 1});

        // Moving sphere
        float x = 5.0f * std::sin(elapsed * 0.5f);
        float z = 5.0f * std::cos(elapsed * 0.5f);
        dd.draw_sphere({x, 2.0f, z}, 1.5f, {0, 1, 1, 1});

        // Lines from origin to sphere
        dd.draw_line({0, 0, 0}, {x, 2.0f, z}, {1, 0.5f, 0, 1});

        // Draw camera frustum (inverse VP lets us visualize the view)
        dd.draw_frustum(glm::inverse(cam.view_projection()), {0.3f, 1, 0.3f, 1});

        dd.render(cam.view_projection());

        window.swap_buffers();
    }

    return EXIT_SUCCESS;
}
