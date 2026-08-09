// Example 10 — gfx::Model: glTF model loading with interactive orbit controls.

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>
#include <gllib/log.hpp>

#include <cstdlib>

static const char* vert_src = R"(
#version 460 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec2 a_uv;
uniform mat4 u_view_proj;
out vec2 v_uv;
out float v_brightness;
void main() {
    gl_Position = u_view_proj * vec4(a_pos, 1.0);
    v_uv = a_uv;
    v_brightness = 0.5 + 0.5 * abs(a_normal.y);
}
)";

static const char* frag_src = R"(
#version 460 core
uniform vec4 u_color;
in vec2 v_uv;
in float v_brightness;
out vec4 frag_color;
void main() {
    frag_color = vec4(u_color.rgb * v_brightness, u_color.a);
}
)";

int main(int argc, char** argv) {
    gllib::log_to_stderr(gllib::LogLevel::info);

    gfx::Window window({"10 Model", 800, 600});
    window.vsync(true);

    // --- Load model ---
    const char* model_path = (argc > 1) ? argv[1] : "cube.gltf";
    gfx::Model model;
    if (!model.load(model_path)) {
        gllib::log(gllib::LogLevel::error, "failed to load model");
        return EXIT_FAILURE;
    }

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

    // Use the model's first material's base color, or default to purple
    glm::vec4 color(0.8f, 0.4f, 0.8f, 1.0f);
    if (model.material_count() > 0) {
        auto& info = model.material_info(0);
        color = {info.base_color_factor[0], info.base_color_factor[1],
                 info.base_color_factor[2], info.base_color_factor[3]};
    }
    mat.set_uniform("u_color", color);

    // --- Camera (orbit target at origin) ---
    gfx::Camera cam;
    cam.perspective(45.0f, float(window.width()) / window.height(), 0.1f, 10000.0f);
    cam.look_at({3, 2, 4}, {0, 0, 0});

    // --- Renderer ---
    gfx::Renderer renderer;
    renderer.set_clear_color(0.05f, 0.05f, 0.1f, 1.0f);
    gl::enable(GL_DEPTH_TEST);

    double prev_mx = 0, prev_my = 0;
    bool mouse_down = false;

    while (!window.should_close()) {
        window.poll_events();

        cam.set_aspect(float(window.width()) / window.height());

        // --- Orbital controls ---
        double mx, my;
        window.cursor_position(mx, my);

        if (window.mouse_down(gfx::MouseButton::left)) {
            if (mouse_down) {
                double dx = mx - prev_mx;
                double dy = my - prev_my;
                cam.orbit(static_cast<float>(dx * 0.05), static_cast<float>(-dy * 0.05));
            }
            mouse_down = true;
        } else {
            mouse_down = false;
        }
        prev_mx = mx;
        prev_my = my;

        // Scroll to zoom
        double scroll = window.scroll_delta();
        if (scroll != 0.0) {
            cam.zoom(static_cast<float>(-scroll * 20.0));
        }

        // --- Draw ---
        renderer.clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        mat.set_uniform("u_view_proj", cam.view_projection());

        for (size_t i = 0; i < model.mesh_count(); ++i) {
            mat.bind();
            model.mesh(i).draw();
        }

        window.swap_buffers();
    }

    return EXIT_SUCCESS;
}
