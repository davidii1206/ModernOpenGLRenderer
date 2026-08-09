// Example 28 — Animation: skinned character with PBR and ImGui controls.

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

int main() {
    gllib::log_to_stderr(gllib::LogLevel::info);
    gfx::Window window({"28 Animation - BrainStem", 1200, 800});

    // --- IBL ---
    gfx::IBLProbe ibl;
    ibl.generate_procedural(256);
    ibl.bake();

    // --- Skybox ---
    gfx::Skybox skybox(ibl.env_map());

    // --- Shadow map ---
    gfx::ShadowMap shadow(2048);

    // --- ImGui ---
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
    if (!model.load("BrainStem.glb")) {
        gllib::log(gllib::LogLevel::error, "Failed to load BrainStem.glb");
        return EXIT_FAILURE;
    }

    gllib::logf(gllib::LogLevel::info,
                "Loaded: %zu meshes, %zu materials, %zu textures, skin=%s, %zu animations",
                model.mesh_count(), model.material_count(), model.texture_count(),
                model.has_skin() ? "yes" : "no", model.animation_count());

    if (!model.has_skin() || model.animation_count() == 0) {
        gllib::log(gllib::LogLevel::error, "Model has no skin or animations");
        return EXIT_FAILURE;
    }

    gfx::Skeleton& skeleton = model.skeleton();
    gfx::AnimationClip& anim = model.animation(0);

    gllib::logf(gllib::LogLevel::info, "Animation: '%s', duration=%.2fs, %zu joints, %zu channels",
                anim.name.c_str(), anim.duration,
                size_t(skeleton.joint_count()), anim.channels.size());

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
    cam.perspective(45.0f, float(window.width()) / window.height(), 0.1f, 100.0f);
    cam.look_at({0, 1, 3}, {0, 1, 0});

    float yaw = 0, pitch = 0, dist = 3.0f;
    double prev_mx = 0, prev_my = 0;
    bool orbiting = false;

    float anim_time = 0.0f;
    float anim_speed = 1.0f;
    bool playing = true;

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

        // Orbit camera (left-mouse drag)
        if (window.mouse_down(gfx::MouseButton::left)) {
            double mx, my;
            window.cursor_position(mx, my);
            if (!orbiting) {
                orbiting = true;
                prev_mx = mx; prev_my = my;
            }
            float dx = float(mx - prev_mx);
            float dy = float(my - prev_my);
            yaw -= dx * 0.005f;
            pitch = glm::clamp(pitch - dy * 0.005f, -1.5f, 1.5f);
            prev_mx = mx; prev_my = my;
        } else {
            orbiting = false;
        }

        // Scroll zoom
        dist -= float(window.scroll_delta()) * 0.5f;
        dist = glm::clamp(dist, 1.0f, 20.0f);

        cam.set_aspect(float(window.width()) / window.height());
        float cy = std::cos(yaw), sy = std::sin(yaw);
        float cp = std::cos(pitch), sp = std::sin(pitch);
        glm::vec3 cam_pos = glm::vec3(sy * cp, sp, cy * cp) * dist;
        cam.look_at(cam_pos, {0, 1, 0});

        // Light
        glm::vec3 light_dir = glm::normalize(glm::vec3(1, -1.5f, 1));
        glm::mat4 light_vp = gfx::compute_light_vp(
            cam.view_projection(), light_dir, 2.0f, 5.0f);

        // --- Animation update ---
        if (playing) {
            anim_time += dt * anim_speed;
            if (anim_time > anim.duration) anim_time = 0.0f;
        }
        anim.sample(anim_time, skeleton);
        skeleton.update();

        // --- Shadow pass ---
        shadow.begin();
        for (size_t idx : opaque) {
            shadow.render_mesh(model.mesh(idx), light_vp, skeleton.palette_ssbo());
        }
        shadow.end();

        // --- Main pass ---
        gl::viewport(0, 0, window.width(), window.height());
        renderer.clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        skybox.render(cam);

        pbr.begin(cam.view(), cam.projection(), cam.position());
        pbr.set_model_matrix(glm::mat4(1.0f));
        pbr.set_ibl(ibl);
        pbr.set_shadow(shadow, light_vp);
        pbr.set_ambient_hemi({0.3f, 0.4f, 0.6f}, {0.1f, 0.08f, 0.06f}, 0.15f);

        // Opaque meshes (skinned)
        pbr.set_skin(skeleton.palette_ssbo());
        for (size_t idx : opaque) {
            int mi = model.mesh_material(int(idx));
            pbr.set_material(model.material_info(size_t(mi >= 0 ? mi : 0)), model);
            pbr.draw(model.mesh(idx));
        }

        // Masked meshes (if any)
        if (!masked.empty()) {
            for (size_t idx : masked) {
                int mi = model.mesh_material(int(idx));
                pbr.set_material(model.material_info(size_t(mi >= 0 ? mi : 0)), model);
                pbr.draw(model.mesh(idx));
            }
        }

        if (!blended.empty()) {
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
        }

        pbr.end();

        // --- ImGui ---
        gui.begin_frame();
        {
            ImGui::Begin("Animation Controls");

            ImGui::Text("Animation: %s", anim.name.c_str());
            ImGui::Text("Duration: %.2f s", anim.duration);
            ImGui::Text("Joints: %d", skeleton.joint_count());
            ImGui::Text("Channels: %zu", anim.channels.size());
            ImGui::Separator();

            if (ImGui::Button(playing ? "Pause" : "Play"))
                playing = !playing;
            ImGui::SameLine();
            ImGui::SliderFloat("Speed", &anim_speed, 0.0f, 3.0f, "%.2fx");

            bool scrubbing = ImGui::SliderFloat("Time", &anim_time, 0.0f, anim.duration, "%.3f s");
            if (scrubbing) {
                playing = false;
                anim.sample(anim_time, skeleton);
                skeleton.update();
            }

            ImGui::Separator();
            ImGui::Text("Joint Hierarchy");
            for (int i = 0; i < skeleton.joint_count(); ++i) {
                const auto& j = skeleton.joint(i);
                if (j.parent >= 0) {
                    ImGui::Text("    %s  \xe2\x86\x92  %s",
                                skeleton.joint(j.parent).name.c_str(), j.name.c_str());
                } else {
                    ImGui::Text("  %s (root)", j.name.c_str());
                }
            }

            ImGui::End();
        }
        gui.render();

        window.swap_buffers();
        window.poll_events();
    }

    return EXIT_SUCCESS;
}
