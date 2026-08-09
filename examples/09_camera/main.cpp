// Example 09 — gfx::Camera: perspective, orbit controls with delta time.

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>

#include <cstdlib>
#include <vector>

static const char* vert_src = R"(
#version 460 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec2 a_uv;
uniform mat4 u_view_proj;
uniform float u_time;
out vec2 v_uv;
out float v_brightness;
void main() {
    float angle = u_time * 0.3;
    float ca = cos(angle), sa = sin(angle);
    vec3 p = vec3(a_pos.x * ca - a_pos.z * sa, a_pos.y, a_pos.x * sa + a_pos.z * ca);
    gl_Position = u_view_proj * vec4(p, 1.0);
    v_uv = a_uv;
    v_brightness = 0.5 + 0.5 * abs(a_normal.y);
}
)";

static const char* frag_src = R"(
#version 460 core
uniform sampler2D u_tex;
in vec2 v_uv;
in float v_brightness;
out vec4 frag_color;
void main() {
    vec4 t = texture(u_tex, v_uv);
    frag_color = vec4(t.rgb * v_brightness, 1.0);
}
)";

static std::vector<unsigned char> make_checkerboard(int w, int h, int grid) {
    std::vector<unsigned char> pixels(static_cast<std::size_t>(w * h * 4));
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            bool white = ((x / grid) + (y / grid)) % 2 == 0;
            auto off = static_cast<std::size_t>((y * w + x) * 4);
            pixels[off + 0] = white ? 220 : 30;
            pixels[off + 1] = white ? 120 : 30;
            pixels[off + 2] = white ? 30 : 220;
            pixels[off + 3] = 255;
        }
    return pixels;
}

int main() {
    gfx::Window window({"09 Camera", 800, 600});
    window.vsync(true);

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

    // --- Texture ---
    auto pixels = make_checkerboard(256, 256, 16);
    auto tex = std::make_shared<gfx::Texture>();
    tex->create(256, 256, GL_RGBA8);
    tex->upload(pixels.data());
    tex->parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    tex->parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // --- Shader ---
    gl::Shader vs(gl::ShaderType::vertex, vert_src);
    if (!vs.compiled()) return EXIT_FAILURE;
    gl::Shader fs(gl::ShaderType::fragment, frag_src);
    if (!fs.compiled()) return EXIT_FAILURE;
    auto prog = std::make_unique<gl::Program>();
    prog->attach(vs);
    prog->attach(fs);
    if (!prog->link()) return EXIT_FAILURE;

    // --- Material ---
    gfx::Material mat;
    mat.set_program(std::move(prog));
    mat.set_texture("u_tex", tex);

    gfx::Camera cam;
    cam.perspective(45.0f, float(window.width()) / window.height(), 0.1f, 100.0f);
    cam.look_at({3, 2, 4}, {0, 0, 0});

    // --- Renderer ---
    gfx::Renderer renderer;
    renderer.set_clear_color(0.05f, 0.05f, 0.1f, 1.0f);
    gl::enable(GL_DEPTH_TEST);

    float last_time = window.time();
    while (!window.should_close()) {
        float now = window.time();
        float dt = now - last_time;
        last_time = now;

        float orbit_speed = 1.5f;
        float zoom_speed = 1.0f;
        if (window.key_down(gfx::Key::left))  cam.orbit(-orbit_speed * dt, 0);
        if (window.key_down(gfx::Key::right)) cam.orbit( orbit_speed * dt, 0);
        if (window.key_down(gfx::Key::up))    cam.orbit( 0, -orbit_speed * dt);
        if (window.key_down(gfx::Key::down))  cam.orbit( 0,  orbit_speed * dt);
        if (window.key_down(gfx::Key::w))     cam.zoom(-zoom_speed * dt);
        if (window.key_down(gfx::Key::s))     cam.zoom( zoom_speed * dt);
        if (window.key_down(gfx::Key::r)) {
            cam.look_at({3, 2, 4}, {0, 0, 0});
        }

        renderer.clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        cam.set_aspect(float(window.width()) / window.height());
        mat.set_uniform("u_view_proj", cam.view_projection());
        mat.set_uniform("u_time", now);

        renderer.draw(mesh, mat);

        window.swap_buffers();
        window.poll_events();
    }

    return EXIT_SUCCESS;
}
