// No GLFW includes — gfx::Window handles everything.

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>

#include <cstdlib>

static const char* vert_src = R"(
#version 460 core
layout (location = 0) in vec2 a_pos;
void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
)";

static const char* frag_src = R"(
#version 460 core
out vec4 frag_color;
void main() {
    frag_color = vec4(0.3, 0.6, 1.0, 1.0);
}
)";

int main() {
    gfx::Window window({"01 Hello Triangle", 800, 600});

    const float verts[] = {
        -0.5f, -0.5f,
         0.5f, -0.5f,
         0.0f,  0.5f,
    };

    gl::Buffer vbo(gl::BufferType::vertex);
    vbo.data(verts, sizeof(verts));

    gl::VertexArray vao;
    vao.bind();
    vbo.bind();
    vao.attrib_pointer(0, 2, GL_FLOAT, false, 0, (void*)0);
    vao.enable_attrib(0);

    gl::Shader vert(gl::ShaderType::vertex, vert_src);
    if (!vert.compiled()) return EXIT_FAILURE;

    gl::Shader frag(gl::ShaderType::fragment, frag_src);
    if (!frag.compiled()) return EXIT_FAILURE;

    gl::Program prog;
    prog.attach(vert);
    prog.attach(frag);
    if (!prog.link()) return EXIT_FAILURE;

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
