// Example 16 — gfx::BindlessManager: bindless texture sampling via ARB_bindless_texture.

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>

#include <cstdlib>
#include <vector>

static const char* vert_src = R"(
#version 460 core
layout (location = 0) in vec2 a_pos;
layout (location = 1) in vec2 a_uv;
layout (location = 2) in float a_tex_index;
out vec2 v_uv;
flat out int v_tex_index;
void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_uv = a_uv;
    v_tex_index = int(a_tex_index);
}
)";

static const char* frag_src = R"(
#version 460 core
#extension GL_ARB_bindless_texture : require

layout(std430, binding = 0) readonly buffer TextureHandles {
    uvec2 tex_handles[];
};

in vec2 v_uv;
flat in int v_tex_index;
out vec4 frag_color;

void main() {
    sampler2D s = sampler2D(tex_handles[v_tex_index]);
    frag_color = texture(s, v_uv);
}
)";

static std::vector<unsigned char> make_checkerboard(int w, int h, int grid,
                                                      unsigned char c1,
                                                      unsigned char c2) {
    std::vector<unsigned char> pixels(static_cast<std::size_t>(w * h * 4));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            bool white = ((x / grid) + (y / grid)) % 2 == 0;
            auto off = static_cast<std::size_t>((y * w + x) * 4);
            if (white) {
                pixels[off + 0] = c1;
                pixels[off + 1] = c1;
                pixels[off + 2] = c1;
            } else {
                pixels[off + 0] = c2;
                pixels[off + 1] = c2;
                pixels[off + 2] = c2;
            }
            pixels[off + 3] = 255;
        }
    }
    return pixels;
}

int main() {
    gfx::Window window({"16 Bindless", 800, 600});

    // Two quads side by side, each referencing a different texture index
    struct Vertex {
        float pos[2];
        float uv[2];
        float tex_index;
    };

    const Vertex verts[] = {
        // Left quad (tex_index = 0)
        {{-0.9f, -0.8f}, {0.0f, 0.0f}, 0.0f},
        {{ 0.0f, -0.8f}, {1.0f, 0.0f}, 0.0f},
        {{ 0.0f,  0.8f}, {1.0f, 1.0f}, 0.0f},
        {{-0.9f,  0.8f}, {0.0f, 1.0f}, 0.0f},
        // Right quad (tex_index = 1)
        {{ 0.0f, -0.8f}, {0.0f, 0.0f}, 1.0f},
        {{ 0.9f, -0.8f}, {1.0f, 0.0f}, 1.0f},
        {{ 0.9f,  0.8f}, {1.0f, 1.0f}, 1.0f},
        {{ 0.0f,  0.8f}, {0.0f, 1.0f}, 1.0f},
    };

    const unsigned int indices[] = {
        0, 1, 2, 0, 2, 3,
        4, 5, 6, 4, 6, 7,
    };

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
    vao.attrib_pointer(2, 1, GL_FLOAT, false, sizeof(Vertex), (void*)offsetof(Vertex, tex_index));
    vao.enable_attrib(2);

    // --- Two procedural textures ---
    auto pixels0 = make_checkerboard(256, 256, 32, 200, 40);
    gfx::Texture tex0;
    tex0.create(256, 256, GL_RGBA8);
    tex0.upload(pixels0.data(), GL_RGBA, GL_UNSIGNED_BYTE);
    tex0.parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    tex0.parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    tex0.parameter(GL_TEXTURE_WRAP_S, GL_REPEAT);
    tex0.parameter(GL_TEXTURE_WRAP_T, GL_REPEAT);

    auto pixels1 = make_checkerboard(256, 256, 16, 60, 180);
    gfx::Texture tex1;
    tex1.create(256, 256, GL_RGBA8);
    tex1.upload(pixels1.data(), GL_RGBA, GL_UNSIGNED_BYTE);
    tex1.parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    tex1.parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    tex1.parameter(GL_TEXTURE_WRAP_S, GL_REPEAT);
    tex1.parameter(GL_TEXTURE_WRAP_T, GL_REPEAT);

    // --- BindlessManager ---
    gfx::BindlessManager bm(4);
    int idx0 = bm.add(tex0);
    int idx1 = bm.add(tex1);
    bm.upload();
    bm.bind(0);

    // --- Shader ---
    gl::Shader vert(gl::ShaderType::vertex, vert_src);
    if (!vert.compiled()) return EXIT_FAILURE;
    gl::Shader frag(gl::ShaderType::fragment, frag_src);
    if (!frag.compiled()) return EXIT_FAILURE;
    gl::Program prog;
    prog.attach(vert);
    prog.attach(frag);
    if (!prog.link()) return EXIT_FAILURE;

    gfx::Renderer renderer;
    renderer.set_clear_color(0.1f, 0.1f, 0.15f, 1.0f);

    while (!window.should_close()) {
        renderer.clear(GL_COLOR_BUFFER_BIT);

        prog.use();
        bm.bind(0);
        vao.bind();
        gl::draw_elements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, 0);

        window.swap_buffers();
        window.poll_events();
    }

    return EXIT_SUCCESS;
}
