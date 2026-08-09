// No GLFW includes — gfx::Window handles everything.

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>

#include <cstdlib>

static const char* vert_src = R"(
#version 460 core
layout (location = 0) in vec2 a_pos;
layout (location = 1) in vec3 a_col;
out vec3 v_col;
void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_col = a_col;
}
)";

static const char* frag_src = R"(
#version 460 core
in vec3 v_col;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_col, 1.0);
}
)";

struct Vertex {
    float pos[2];
    float col[3];
};

static const Vertex vertices[] = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{ 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{ 0.0f,  0.5f}, {0.0f, 0.0f, 1.0f}},
};

int main() {
    gfx::Window window({"03 Buffer Test", 800, 600});

    gl::Buffer vbo(gl::BufferType::vertex);
    vbo.data(vertices, sizeof(vertices));

    gl::VertexArray vao;
    vao.bind();
    vbo.bind();
    vao.attrib_pointer(0, 2, GL_FLOAT, false, sizeof(Vertex), (void*)offsetof(Vertex, pos));
    vao.enable_attrib(0);
    vao.attrib_pointer(1, 3, GL_FLOAT, false, sizeof(Vertex), (void*)offsetof(Vertex, col));
    vao.enable_attrib(1);

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
