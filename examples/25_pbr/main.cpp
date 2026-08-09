// Example 25 — PBR / IBL: image-based lighting with PBR model rendering.

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>
#include <gfx/skybox.hpp>
#include <gllib/log.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>
#include <gfx/imgui_overlay.hpp>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------------------
// FPS camera controls
// ---------------------------------------------------------------------------

static void fps_control(gfx::Window& window, gfx::Camera& cam,
                        float dt, float& yaw, float& pitch, bool& captured) {
    static double prev_x = 0, prev_y = 0;
    static bool first = true;
    float speed = 3.0f * dt;
    float sens = 0.002f;

    if (window.mouse_down(gfx::MouseButton::right)) {
        if (!captured) {
            glfwSetInputMode((GLFWwindow*)window.native_handle(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            captured = true;
            first = true;
        }
    } else if (captured) {
        glfwSetInputMode((GLFWwindow*)window.native_handle(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        captured = false;
    }

    if (captured) {
        double cx, cy;
        window.cursor_position(cx, cy);
        if (first) { prev_x = cx; prev_y = cy; first = false; }
        yaw -= float(cx - prev_x) * sens;
        pitch = glm::clamp(pitch + float(prev_y - cy) * sens, -1.5f, 1.5f);
        prev_x = cx;
        prev_y = cy;
    }

    float cy = std::cos(yaw), sy = std::sin(yaw);
    glm::vec3 fwd(sy, 0, cy), rgt(cy, 0, -sy);
    glm::vec3 vel(0);
    if (window.key_down(gfx::Key::w)) vel += fwd;
    if (window.key_down(gfx::Key::s)) vel -= fwd;
    if (window.key_down(gfx::Key::a)) vel += rgt;
    if (window.key_down(gfx::Key::d)) vel -= rgt;
    if (window.key_down(gfx::Key::space)) vel.y += 1;
    if (window.key_down(gfx::Key::shift)) vel.y -= 1;
    if (glm::length(vel) > 0.0f) vel = glm::normalize(vel) * speed;

    glm::vec3 pos = cam.position() + vel;
    glm::vec3 dir(std::cos(pitch) * sy, std::sin(pitch), std::cos(pitch) * cy);
    cam.look_at(pos, pos + dir);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    gllib::log_to_stderr(gllib::LogLevel::info);
    gfx::Window window({"25 PBR / IBL - Sponza", 1200, 800});

    // --- IBL ---
    gfx::IBLProbe ibl;
    ibl.generate_procedural(256);
    ibl.bake();

    // --- Skybox ---
    gfx::Skybox skybox(ibl.env_map());

    // --- Shadow map ---
    gfx::ShadowMap shadow(8192);

    // --- ImGui overlay ---
    gfx::ImGuiOverlay gui;
    if (!gui.init(window)) {
        gllib::log(gllib::LogLevel::error, "ImGui init failed");
        return EXIT_FAILURE;
    }

    // --- PBR material ---
    gfx::PBRMaterial pbr;
    if (!pbr.valid()) {
        gllib::log(gllib::LogLevel::error, "PBR program failed to compile/link");
        return EXIT_FAILURE;
    }

    // --- Load model ---
    gfx::Model model;
    if (!model.load("sponza.glb")) {
        gllib::log(gllib::LogLevel::error, "Failed to load Sponza model");
        return EXIT_FAILURE;
    }
    gllib::logf(gllib::LogLevel::info,
                "Loaded Sponza: %zu meshes, %zu materials, %zu textures",
                model.mesh_count(), model.material_count(), model.texture_count());

    // Compute scene bounds from bounding spheres (applied 0.01 model scale)
    glm::vec3 scene_min(FLT_MAX), scene_max(-FLT_MAX);
    for (size_t i = 0; i < model.mesh_count(); ++i) {
        glm::vec4 bs = model.mesh_bounding_sphere(int(i));
        glm::vec3 center = glm::vec3(bs) * 0.01f;
        float r = bs.w * 0.01f;
        for (int j = 0; j < 3; ++j) {
            scene_min[j] = std::min(scene_min[j], center[j] - r);
            scene_max[j] = std::max(scene_max[j], center[j] + r);
        }
    }
    float scene_radius = glm::length(scene_max - scene_min) * 0.5f;

    // Sort meshes by alpha mode
    std::vector<size_t> opaque, masked, blended;
    for (size_t i = 0; i < model.mesh_count(); ++i) {
        int mi = model.mesh_material(int(i));
        auto& mat = model.material_info(size_t(mi >= 0 ? mi : 0));
        switch (mat.alpha_mode) {
            case gfx::AlphaMode_Mask:  masked.push_back(i); break;
            case gfx::AlphaMode_Blend: blended.push_back(i); break;
            default:                   opaque.push_back(i); break;
        }
    }

    // --- Camera ---
    gfx::Camera cam;
    cam.perspective(45.0f, float(window.width()) / window.height(), 0.1f, 1000.0f);
    cam.look_at({0, 1.5f, 5}, {0, 1.5f, 4});

    glm::mat4 model_mat = glm::scale(glm::mat4(1.0f), glm::vec3(0.01f));
    glm::vec3 light_dir = glm::normalize(glm::vec3(1, -1.5f, 1));

    float yaw = 0, pitch = 0;
    bool captured = false;

    gfx::Renderer renderer;
    renderer.set_clear_color(0.08f, 0.08f, 0.10f, 1.0f);
    gl::enable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

    double last = window.time();

    while (!window.should_close()) {
        double now = window.time();
        float dt = float(now - last);
        last = now;

        window.poll_events();
        fps_control(window, cam, dt, yaw, pitch, captured);
        cam.set_aspect(float(window.width()) / window.height());

        // Compute light VP that covers the scene
        glm::mat4 light_vp = gfx::compute_light_vp(
            cam.view_projection(), light_dir, scene_radius, 10.0f);

        // --- Shadow pass ---
        shadow.begin();
        for (size_t idx : opaque) {
            glm::mat4 mvp = light_vp * model_mat;
            shadow.render_mesh(model.mesh(idx), mvp);
        }
        for (size_t idx : masked) {
            glm::mat4 mvp = light_vp * model_mat;
            shadow.render_mesh(model.mesh(idx), mvp);
        }
        shadow.end();

        // --- Main pass ---
        gl::viewport(0, 0, window.width(), window.height());
        renderer.clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Skybox
        skybox.render(cam);

        pbr.begin(cam.view(), cam.projection(), cam.position());
        pbr.set_model_matrix(model_mat);
        pbr.set_ibl(ibl);
        pbr.set_shadow(shadow, light_vp);
        pbr.set_ambient_hemi({0.3f, 0.4f, 0.6f}, {0.1f, 0.08f, 0.06f}, 0.15f);

        // Opaque
        for (size_t idx : opaque) {
            int mi = model.mesh_material(int(idx));
            pbr.set_material(model.material_info(size_t(mi >= 0 ? mi : 0)), model);
            pbr.draw(model.mesh(idx));
        }

        // Masked
        for (size_t idx : masked) {
            int mi = model.mesh_material(int(idx));
            pbr.set_material(model.material_info(size_t(mi >= 0 ? mi : 0)), model);
            pbr.draw(model.mesh(idx));
        }

        // Blended
        gl::enable(GL_BLEND);
        gl::blend_func(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        gl::depth_mask(GL_FALSE);
        for (size_t idx : blended) {
            int mi = model.mesh_material(int(idx));
            pbr.set_material(model.material_info(size_t(mi >= 0 ? mi : 0)), model);
            pbr.draw(model.mesh(idx));
        }
        gl::depth_mask(GL_TRUE);
        gl::disable(GL_BLEND);

        pbr.end();

        gui.begin_frame();
        {
            ImGui::Begin("Shadow Map");
            float sz = float(ImGui::GetContentRegionAvail().x);
            if (sz < 64) sz = 512;
            ImGui::Image((ImTextureID)(intptr_t)shadow.handle(), ImVec2(sz, sz));
            ImGui::Text("Size: %d x %d", shadow.size(), shadow.size());
            ImGui::Text("Scene radius: %.3f", scene_radius);
            ImGui::End();
        }
        gui.render();

        window.swap_buffers();
        window.poll_events();
    }

    return EXIT_SUCCESS;
}
