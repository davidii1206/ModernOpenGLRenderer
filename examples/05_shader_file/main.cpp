// Example 05 — Loading shaders from files using gl::shader_from_file / gl::program_from_files.

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>

#include <cstdlib>

struct Vertex {
    float pos[2];
    float col[3];
};

int main() {
    gfx::Window window({"05 Shader File", 800, 600});
    window.vsync(false);

    auto prog = gl::program_from_files("vert.glsl", "frag.glsl");
    if (!prog.linked()) return EXIT_FAILURE;

    const Vertex verts[] = {
        {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
        {{ 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
        {{ 0.0f,  0.5f}, {0.0f, 0.0f, 1.0f}},
    };

    gl::Buffer vbo(gl::BufferType::vertex);
    vbo.data(verts, sizeof(verts));

    gl::VertexArray vao;
    vao.bind();
    vbo.bind();
    vao.attrib_pointer(0, 2, GL_FLOAT, false, sizeof(Vertex), (void*)offsetof(Vertex, pos));
    vao.enable_attrib(0);
    vao.attrib_pointer(1, 3, GL_FLOAT, false, sizeof(Vertex), (void*)offsetof(Vertex, col));
    vao.enable_attrib(1);

    while (!window.should_close()) {
        gl::clear(GL_COLOR_BUFFER_BIT);

        prog.use();
        vao.bind();
        gl::draw_arrays(GL_TRIANGLES, 0, 3);

        window.swap_buffers();
        window.poll_events();
    }

    return EXIT_SUCCESS;
}
