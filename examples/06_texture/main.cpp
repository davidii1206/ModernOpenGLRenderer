// Example 06 — gfx::Texture: create procedural checkerboard, display on a quad.

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>

#include <cstdlib>
#include <vector>

static const char* vert_src = R"(
#version 460 core
layout (location = 0) in vec2 a_pos;
layout (location = 1) in vec2 a_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_uv = a_uv;
}
)";

static const char* frag_src = R"(
#version 460 core
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 frag_color;
void main() {
    frag_color = texture(u_tex, v_uv);
}
)";

// Generate a checkerboard pattern
static std::vector<unsigned char> make_checkerboard(int w, int h, int grid) {
    std::vector<unsigned char> pixels(static_cast<std::size_t>(w * h * 4));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            bool white = ((x / grid) + (y / grid)) % 2 == 0;
            auto off = static_cast<std::size_t>((y * w + x) * 4);
            pixels[off + 0] = white ? 255 : 30;
            pixels[off + 1] = white ? 255 : 30;
            pixels[off + 2] = white ? 255 : 30;
            pixels[off + 3] = 255;
        }
    }
    return pixels;
}

struct Vertex {
    float pos[2];
    float uv[2];
};

int main() {
    gfx::Window window({"06 Texture", 800, 600});

    // --- Textured quad mesh ---
    const Vertex verts[] = {
        {{-0.8f, -0.8f}, {0.0f, 0.0f}},
        {{ 0.8f, -0.8f}, {1.0f, 0.0f}},
        {{ 0.8f,  0.8f}, {1.0f, 1.0f}},
        {{-0.8f,  0.8f}, {0.0f, 1.0f}},
    };
    const unsigned int indices[] = {0, 1, 2, 0, 2, 3};

    gl::Buffer vbo(gl::BufferType::vertex);
    vbo.data(verts, sizeof(verts));
    gl::Buffer ebo(gl::BufferType::index);
    ebo.data(indices, sizeof(indices));

    gl::VertexArray vao;
    vao.bind();
    vbo.bind();
    ebo.bind();
    vao.attrib_pointer(0, 2, GL_FLOAT, false, sizeof(Vertex), (void*)offsetof(Vertex, pos));
    vao.enable_attrib(0);
    vao.attrib_pointer(1, 2, GL_FLOAT, false, sizeof(Vertex), (void*)offsetof(Vertex, uv));
    vao.enable_attrib(1);

    // --- gfx::Texture (no raw GL calls) ---
    auto pixels = make_checkerboard(512, 512, 32);
    gfx::Texture tex;
    tex.create(512, 512, GL_RGBA8);
    tex.upload(pixels.data(), GL_RGBA, GL_UNSIGNED_BYTE);
    tex.parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    tex.parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    tex.parameter(GL_TEXTURE_WRAP_S, GL_REPEAT);
    tex.parameter(GL_TEXTURE_WRAP_T, GL_REPEAT);
    tex.generate_mipmap();

    // --- Shader program ---
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
        tex.bind(0);
        prog.uniform1i(prog.uniform_location("u_tex"), 0);
        vao.bind();
        gl::draw_elements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        window.swap_buffers();
        window.poll_events();
    }

    return EXIT_SUCCESS;
}
