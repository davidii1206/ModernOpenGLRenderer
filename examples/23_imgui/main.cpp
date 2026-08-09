// Example 23 — Dear ImGui overlay via gfx::ImGuiOverlay

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>
#include <imgui.h>

#include <cstdlib>
#include <cstdio>
#include <cmath>

static const char* vert_src = R"(
#version 460 core
layout (location = 0) in vec2 a_pos;
layout (location = 1) in vec3 a_color;
uniform mat4 u_transform = mat4(1);
out vec3 v_color;
void main() {
    v_color = a_color;
    gl_Position = u_transform * vec4(a_pos, 0.0, 1.0);
}
)";

static const char* frag_src = R"(
#version 460 core
in vec3 v_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_color, 1.0);
}
)";

int main() {
    gfx::Window window({"23 ImGui Overlay", 800, 600});
    window.vsync(false);

    // Triangle with per-vertex colors
    const float verts[] = {
        -0.5f, -0.5f,  1.0f, 0.2f, 0.2f,
         0.5f, -0.5f,  0.2f, 1.0f, 0.2f,
         0.0f,  0.5f,  0.2f, 0.2f, 1.0f,
    };

    gl::Buffer vbo(gl::BufferType::vertex);
    vbo.data(verts, sizeof(verts));

    gl::VertexArray vao;
    vao.bind();
    vbo.bind();
    vao.attrib_pointer(0, 2, GL_FLOAT, false, 5 * sizeof(float), (void*)0);
    vao.enable_attrib(0);
    vao.attrib_pointer(1, 3, GL_FLOAT, false, 5 * sizeof(float), (void*)(2 * sizeof(float)));
    vao.enable_attrib(1);

    gl::Shader vert(gl::ShaderType::vertex, vert_src);
    if (!vert.compiled()) return EXIT_FAILURE;
    gl::Shader frag(gl::ShaderType::fragment, frag_src);
    if (!frag.compiled()) return EXIT_FAILURE;

    gl::Program prog;
    prog.attach(vert);
    prog.attach(frag);
    if (!prog.link()) return EXIT_FAILURE;

    // ImGui overlay
    gfx::ImGuiOverlay gui;
    if (!gui.init(window)) {
        std::fprintf(stderr, "ImGuiOverlay init failed\n");
        return EXIT_FAILURE;
    }

    float angle = 0.0f;
    float clear_color[4] = {0.08f, 0.08f, 0.12f, 1.0f};
    bool show_demo = false;

    while (!window.should_close()) {
        window.poll_events();

        angle += 0.02f;
        if (angle > 6.2832f) angle -= 6.2832f;

        gui.begin_frame();
        {
            ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
            ImGui::Begin("ImGui Example", nullptr, ImGuiWindowFlags_NoSavedSettings);
            ImGui::Text("FPS: %.0f", ImGui::GetIO().Framerate);
            ImGui::Separator();
            ImGui::ColorEdit3("Clear Color", clear_color);
            ImGui::SliderFloat("Rotation", &angle, 0.0f, 6.2832f);
            ImGui::Checkbox("Show Demo Window", &show_demo);
            if (ImGui::Button("Quit"))
                break;
            ImGui::End();

            if (show_demo)
                ImGui::ShowDemoWindow(&show_demo);
        }

        gl::clear_color(clear_color[0], clear_color[1], clear_color[2], clear_color[3]);
        gl::clear(GL_COLOR_BUFFER_BIT);

        // Rotation matrix around Z
        float c = std::cos(angle);
        float s = std::sin(angle);
        float rot[16] = {
            c, 0, -s, 0,
            0, 1,  0, 0,
            s, 0,  c, 0,
            0, 0,  0, 1,
        };
        GLint u_transform = glGetUniformLocation(prog.handle(), "u_transform");
        glUniformMatrix4fv(u_transform, 1, GL_FALSE, rot);

        prog.use();
        vao.bind();
        gl::draw_arrays(GL_TRIANGLES, 0, 3);

        gui.render();
        window.swap_buffers();
    }

    return EXIT_SUCCESS;
}
