// Example 07 — gfx::Mesh + gfx::Material: exploded textured cube with camera.

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>

#include <cstdlib>
#include <vector>

static const char* vert_src = R"(
#version 460 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec2 a_uv;
layout (location = 3) in vec4 a_tangent;
uniform mat4 u_view_proj;
out vec2 v_uv;
void main() {
    gl_Position = u_view_proj * vec4(a_pos, 1.0);
    v_uv = a_uv;
}
)";

static const char* frag_src = R"(
#version 460 core
uniform sampler2D u_tex;
uniform float u_time;
in vec2 v_uv;
out vec4 frag_color;
void main() {
    vec2 uv = v_uv + vec2(u_time * 0.05);
    frag_color = texture(u_tex, uv);
}
)";

static std::vector<unsigned char> make_checkerboard(int w, int h, int grid) {
    std::vector<unsigned char> pixels(static_cast<std::size_t>(w * h * 4));
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            bool white = ((x / grid) + (y / grid)) % 2 == 0;
            auto off = static_cast<std::size_t>((y * w + x) * 4);
            pixels[off + 0] = white ? 200 : 40;
            pixels[off + 1] = white ? 100 : 40;
            pixels[off + 2] = white ? 200 : 80;
            pixels[off + 3] = 255;
        }
    return pixels;
}

int main() {
    gfx::Window window({"07 Mesh + Material", 800, 600});

    // --- Cube mesh ---
    std::vector<gfx::Vertex> verts;
    std::vector<unsigned int> idx;

    auto face = [&](float cx, float cy, float cz, float nx, float ny, float nz,
                    float ux, float uy, float uz, float vx, float vy, float vz) {
        auto base = static_cast<unsigned int>(verts.size());
        float h = 1.0f;
        verts.push_back({{cx - ux*h - vx*h, cy - uy*h - vy*h, cz - uz*h - vz*h}, {nx, ny, nz}, {0, 0}, {ux, uy, uz, 0}});
        verts.push_back({{cx + ux*h - vx*h, cy + uy*h - vy*h, cz + uz*h - vz*h}, {nx, ny, nz}, {1, 0}, {ux, uy, uz, 0}});
        verts.push_back({{cx + ux*h + vx*h, cy + uy*h + vy*h, cz + uz*h + vz*h}, {nx, ny, nz}, {1, 1}, {ux, uy, uz, 0}});
        verts.push_back({{cx - ux*h + vx*h, cy - uy*h + vy*h, cz - uz*h + vz*h}, {nx, ny, nz}, {0, 1}, {ux, uy, uz, 0}});
        idx.insert(idx.end(), {base, base+1, base+2, base, base+2, base+3});
    };

    face( 0, 0, 1,  0, 0, 1,  1, 0, 0,  0, 1, 0);
    face( 0, 0,-1,  0, 0,-1, -1, 0, 0,  0, 1, 0);
    face( 1, 0, 0,  1, 0, 0,  0, 0,-1,  0, 1, 0);
    face(-1, 0, 0, -1, 0, 0,  0, 0, 1,  0, 1, 0);
    face( 0, 1, 0,  0, 1, 0,  1, 0, 0,  0, 0, 1);
    face( 0,-1, 0,  0,-1, 0,  1, 0, 0,  0, 0,-1);

    gfx::Mesh mesh;
    mesh.set_vertices(verts);
    mesh.set_indices(idx);
    mesh.build();

    // --- Checkerboard texture ---
    auto pixels = make_checkerboard(256, 256, 16);
    auto tex = std::make_shared<gfx::Texture>();
    tex->create(256, 256, GL_RGBA8);
    tex->upload(pixels.data());
    tex->parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    tex->parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // --- Shader program ---
    gl::Shader vert_shader(gl::ShaderType::vertex, vert_src);
    if (!vert_shader.compiled()) return EXIT_FAILURE;
    gl::Shader frag_shader(gl::ShaderType::fragment, frag_src);
    if (!frag_shader.compiled()) return EXIT_FAILURE;
    auto prog = std::make_unique<gl::Program>();
    prog->attach(vert_shader);
    prog->attach(frag_shader);
    if (!prog->link()) return EXIT_FAILURE;

    // --- Camera ---
    gfx::Camera cam;
    cam.perspective(45.0f, float(window.width()) / window.height(), 0.1f, 100.0f);
    cam.look_at({3, 2, 4}, {0, 0, 0});

    // --- Material ---
    gfx::Material mat;
    mat.set_program(std::move(prog));
    mat.set_texture("u_tex", tex);

    gl::enable(GL_DEPTH_TEST);

    while (!window.should_close()) {
        gl::clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        cam.set_aspect(float(window.width()) / window.height());
        mat.set_uniform("u_view_proj", cam.view_projection());
        mat.set_uniform("u_time", window.time());
        mat.bind();
        mesh.draw();

        window.swap_buffers();
        window.poll_events();
    }

    return EXIT_SUCCESS;
}
