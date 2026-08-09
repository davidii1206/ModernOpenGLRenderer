// Example 11 — gl::Sampler: separate sampler objects with different filtering/wrap modes.

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>

#include <glm/gtc/matrix_transform.hpp>

#include <cstdlib>
#include <vector>

static const char* vert_src = R"(
#version 460 core
layout (location = 0) in vec3 a_pos;
layout (location = 2) in vec2 a_uv;
uniform mat4 u_view_proj;
uniform mat4 u_model;
out vec2 v_uv;
void main() {
    gl_Position = u_view_proj * u_model * vec4(a_pos, 1.0);
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

static std::vector<unsigned char> make_checkerboard(int w, int h, int grid) {
    std::vector<unsigned char> pixels(static_cast<std::size_t>(w * h * 4));
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            bool white = ((x / grid) + (y / grid)) % 2 == 0;
            auto off = static_cast<std::size_t>((y * w + x) * 4);
            pixels[off + 0] = white ? 220 : 30;
            pixels[off + 1] = white ? 220 : 30;
            pixels[off + 2] = white ? 30 : 220;
            pixels[off + 3] = 255;
        }
    return pixels;
}

static std::vector<gfx::Vertex> make_quad() {
    return {
        {{-1, -1, 0}, {0, 0, 1}, {-0.3f, -0.3f}, {}},
        {{ 1, -1, 0}, {0, 0, 1}, { 1.3f, -0.3f}, {}},
        {{ 1,  1, 0}, {0, 0, 1}, { 1.3f,  1.3f}, {}},
        {{-1,  1, 0}, {0, 0, 1}, {-0.3f,  1.3f}, {}},
    };
}

int main() {
    gfx::Window window({"11 Sampler", 800, 600});

    // --- Texture ---
    auto pixels = make_checkerboard(256, 256, 32);
    auto tex = std::make_shared<gfx::Texture>();
    tex->create(256, 256, GL_RGBA8);
    tex->upload(pixels.data());
    // Set base texture params so texture is complete (samplers override per-draw)
    tex->parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    tex->parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // --- Samplers ---
    gl::Sampler samplers[4];
    samplers[0].parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    samplers[0].parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    samplers[0].parameter(GL_TEXTURE_WRAP_S, GL_REPEAT);
    samplers[0].parameter(GL_TEXTURE_WRAP_T, GL_REPEAT);
    samplers[1].parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    samplers[1].parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    samplers[1].parameter(GL_TEXTURE_WRAP_S, GL_REPEAT);
    samplers[1].parameter(GL_TEXTURE_WRAP_T, GL_REPEAT);
    samplers[2].parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    samplers[2].parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    samplers[2].parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    samplers[2].parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    samplers[3].parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    samplers[3].parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    samplers[3].parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    samplers[3].parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // --- Quad mesh ---
    auto verts = make_quad();
    std::vector<unsigned int> idx = {0, 1, 2, 0, 2, 3};
    gfx::Mesh quad;
    quad.set_vertices(verts);
    quad.set_indices(idx);
    quad.build();

    // --- Shader ---
    gl::Shader vs(gl::ShaderType::vertex, vert_src);
    if (!vs.compiled()) return EXIT_FAILURE;
    gl::Shader fs(gl::ShaderType::fragment, frag_src);
    if (!fs.compiled()) return EXIT_FAILURE;
    auto prog = std::make_unique<gl::Program>();
    prog->attach(vs);
    prog->attach(fs);
    if (!prog->link()) return EXIT_FAILURE;

    gfx::Material mat;
    mat.set_program(std::move(prog));
    mat.set_texture("u_tex", tex);

    gfx::Camera cam;
    float aspect = float(window.width()) / window.height();
    cam.ortho_2d(4.0f, aspect);

    gfx::Renderer renderer;
    renderer.set_clear_color(0.1f, 0.1f, 0.1f, 1.0f);

    while (!window.should_close()) {
        renderer.clear(GL_COLOR_BUFFER_BIT);
        cam.set_aspect(float(window.width()) / window.height());

        for (int i = 0; i < 4; ++i) {
            samplers[i].bind(0);

            float x = float(i % 2) * 2.0f - 1.0f;
            float y = float(i / 2) * 2.0f - 1.0f;
            glm::mat4 model = glm::translate(glm::mat4(1), glm::vec3(x * 0.9f, -y * 0.9f, 0))
                            * glm::scale(glm::mat4(1), glm::vec3(0.85f, 0.85f, 1));

            mat.set_uniform("u_view_proj", cam.view_projection());
            mat.set_uniform("u_model", model);
            renderer.draw(quad, mat);
        }

        window.swap_buffers();
        window.poll_events();
    }

    return EXIT_SUCCESS;
}
