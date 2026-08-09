// No GLFW includes — gfx::Window handles everything.

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>

#include <cstdlib>

int main() {
    gfx::Window window({"02 Clear Screen", 800, 600});

    gl::clear_color(0.1f, 0.2f, 0.4f, 1.0f);

    while (!window.should_close()) {
        gl::clear(GL_COLOR_BUFFER_BIT);
        window.swap_buffers();
        window.poll_events();
    }

    return EXIT_SUCCESS;
}
