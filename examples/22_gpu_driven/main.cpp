#include <gl/gl.hpp>
#include <gfx/gfx.hpp>
#include <gfx/fsr.hpp>
#include <imgui.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>

#include <cstdlib>
#include <cstdio>
#include <vector>
#include <string>
#include <regex>
#include <set>
#include <fstream>
#include <filesystem>

#include <gllib/log.hpp>

#include <cmath>

// Simple textual include resolver — no preprocessor branch evaluation.
// The GLSL compiler handles #if/#ifdef/#endif itself.
static std::string read_file_str(const std::string& path) {
    std::ifstream f(path, std::ios::in | std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto size = f.tellg();
    std::string s(static_cast<std::size_t>(size), '\0');
    f.seekg(0);
    f.read(s.data(), size);
    return s;
}

static std::string resolve_glsl_includes(const std::string& source_path,
    const std::vector<std::filesystem::path>& search_dirs,
    std::set<std::string>& included)
{
    std::string src = read_file_str(source_path);
    if (src.empty()) return src;

    static const std::regex re(R"(#\s*include\s+\"([^\"]+)\")");
    std::string result;
    std::smatch m;
    auto it = src.cbegin();
    auto end = src.cend();

    while (std::regex_search(it, end, m, re)) {
        result.append(it, m[0].first);
        std::string inc_name = m[1].str();

        if (!included.insert(inc_name).second) {
            it = m[0].second;
            continue;
        }

        bool found = false;
        for (auto& dir : search_dirs) {
            auto p = dir / inc_name;
            auto content = read_file_str(p.string());
            if (!content.empty()) {
                result += resolve_glsl_includes(p.string(), search_dirs, included);
                result += "\n";
                found = true;
                break;
            }
        }
        if (!found) {
            std::fprintf(stderr, "WARNING: include not found: %s\n", inc_name.c_str());
            result.append(m[0].first, m[0].second);
        }
        it = m[0].second;
    }
    result.append(it, end);
    return result;
}

static std::string resolve_glsl_includes(const std::string& source_path,
    const std::vector<std::filesystem::path>& search_dirs)
{
    std::set<std::string> included;
    return resolve_glsl_includes(source_path, search_dirs, included);
}

int main() {

    gfx::WindowDesc wdesc;
    wdesc.title = "22 GPU-Driven Pipeline";
    wdesc.width = 1024;
    wdesc.height = 768;
    wdesc.debug = true;
    gfx::Window window(wdesc);
    window.vsync(false);

    gllib::log_to_stderr();

    gl::enable_debug_output();

    const char* vendor_str = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
    const char* renderer_str = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    const char* version_str  = reinterpret_cast<const char*>(glGetString(GL_VERSION));
    std::printf("GPU: %s — %s (%s)\n", vendor_str, renderer_str, version_str);

    // Load three models with LODs
    gfx::Model bunny, rocks, diffrock;
    if (!bunny.load("Stanford_Bunny.glb")) {
        std::fprintf(stderr, "Failed to load Stanford_Bunny.glb\n");
        return EXIT_FAILURE;
    }
    if (!rocks.load("Rocks.glb")) {
        std::fprintf(stderr, "Failed to load Rocks.glb\n");
        return EXIT_FAILURE;
    }
    if (!diffrock.load("DifferentRock.glb")) {
        std::fprintf(stderr, "Failed to load DifferentRock.glb\n");
        return EXIT_FAILURE;
    }

    gfx::Model* models[] = {&bunny, &rocks, &diffrock};
    const char* model_names[] = {"Bunny", "Rocks", "DifferentRock"};
    int num_models = 3;

    // Print LOD info
    for (int m = 0; m < num_models; ++m) {
        const auto& lg = models[m]->lod_group(0);
        std::printf("Model %d (%s) LOD group '%s': %zu levels\n",
            m, model_names[m], lg.name.c_str(), lg.mesh_indices.size());
        for (size_t i = 0; i < lg.mesh_indices.size(); ++i) {
            const auto& mesh = models[m]->mesh(lg.mesh_indices[i]);
            std::printf("  LOD %zu: %zu tris, %zu verts\n",
                i, mesh.index_count() / 3, mesh.vertex_count());
        }
    }

    // Hot-reloadable shaders from external files
    gl::HotReloadProgram hot_prog;
    hot_prog.add_stage("vert.glsl", gl::ShaderType::vertex);
    hot_prog.add_stage("frag.glsl", gl::ShaderType::fragment);
    if (!hot_prog.poll()) return EXIT_FAILURE;

    gfx::Material mat;
    mat.set_program(hot_prog.take_program());

    // Quality settings
    int quality_tier = 1; // 0=Low, 1=Medium, 2=High, 3=Ultra
    float render_scale = 0.5f;
    float rcas_sharpness = 0.5f;
    const char* quality_names[] = {"Low", "Medium", "High", "Ultra"};
    const float quality_scales[] = {0.33f, 0.5f, 0.75f, 1.0f};
    const float quality_sharpness[] = {0.8f, 0.5f, 0.5f, 0.2f};

    // Dynamic Resolution Scaling
    bool drs_enabled = false;
    float drs_target_ms = 2.0f;
    float drs_smoothed_ms = 2.0f;

    // Camera
    gfx::Camera cam;
    cam.perspective(60.0f, float(window.width()) / window.height(), 0.1f, 500.0f);
    cam.look_at({40, 30, 50}, {0, 0, 0});

    // GPU pipeline — register models, then finalize
    const int N = 5000;
    gfx::GpuPipeline gpu(N);

    int bunny_h    = gpu.add_model(bunny, 0);
    int rocks_h    = gpu.add_model(rocks, 0);
    int diffrock_h = gpu.add_model(diffrock, 0);
    gpu.finalize();

    // Pre-compute per-slot tri count
    int slot_tris[gfx::GpuPipeline::kMaxSlots] = {};
    for (int s = 0; s < gpu.slot_count(); ++s)
        slot_tris[s] = gpu.slot_index_count(s) / 3;

    // Uniform scaling — all models match bunny's world-space size
    const auto& lg0 = bunny.lod_group(0);
    float bunny_radius = bunny.mesh_bounding_sphere(lg0.mesh_indices[0]).w;
    float bunny_target_scale = 7.0f;
    float bunny_world_radius = bunny_radius * bunny_target_scale;
    float rock_scale   = bunny_world_radius / rocks.mesh_bounding_sphere(rocks.lod_group(0).mesh_indices[0]).w;
    float diff_scale   = bunny_world_radius / diffrock.mesh_bounding_sphere(diffrock.lod_group(0).mesh_indices[0]).w;
    float scales[] = {bunny_target_scale, rock_scale, diff_scale};
    int handles[] = {bunny_h, rocks_h, diffrock_h};
    std::printf("Scales: bunny=%.1f rocks=%.4f diffrock=%.4f  world_radius=%.2f  spacing=%.2f\n",
        scales[0], scales[1], scales[2], bunny_world_radius, bunny_world_radius * 2.5f);

    // Pre-compute all transforms for the grid
    float spacing = bunny_world_radius * 2.5f;
    int per_side = static_cast<int>(ceil(sqrt(static_cast<double>(N))));
    struct Inst { int handle; glm::mat4 transform; };
    std::vector<Inst> all_instances;
    all_instances.reserve(N);
    for (int z = -per_side/2; z < per_side/2 && (int)all_instances.size() < N; ++z) {
        for (int x = -per_side/2; x < per_side/2 && (int)all_instances.size() < N; ++x) {
            int model_id = (int)all_instances.size() % num_models;
            glm::mat4 m = glm::translate(glm::mat4(1), glm::vec3(x, 0, z) * spacing);
            m = glm::rotate(m, glm::radians(90.0f), glm::vec3(1, 0, 0));
            m = glm::scale(m, glm::vec3(scales[model_id]));
            all_instances.push_back({handles[model_id], m});
        }
    }
    std::printf("Total instances: %zu\n", all_instances.size());

    // LOD thresholds (max pixel size for each coarser LOD)
    float lod_thresholds[] = { 200.0f, 100.0f, 50.0f, 25.0f, 12.0f, 6.0f };
    gpu.set_lod_thresholds(lod_thresholds, 6);

    // ─── FSR1 compute programs (EASU upscale + RCAS sharpen) ──────────────
    // Search dirs: dir of the .glsl file, plus the gllib/glsl headers dir
    std::vector<std::filesystem::path> fsr_include_dirs;
    fsr_include_dirs.push_back(std::filesystem::current_path());
    {
        // Find gllib/glsl relative to the executable path
        std::filesystem::path exe_dir = std::filesystem::current_path();
        // Walk up looking for gllib/glsl
        std::filesystem::path probe = exe_dir;
        while (!probe.empty()) {
            auto candidate = probe / "gllib" / "glsl";
            if (std::filesystem::is_directory(candidate)) {
                fsr_include_dirs.push_back(candidate);
                break;
            }
            probe = probe.parent_path();
        }
    }

    auto compile_fsr_program = [&](const char* filename) -> GLuint {
        auto src = resolve_glsl_includes(filename, fsr_include_dirs);
        if (src.empty()) {
            std::fprintf(stderr, "Failed to read %s\n", filename);
            return 0;
        }
        const char* srcs[] = {src.c_str()};
        GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
        glShaderSource(shader, 1, srcs, nullptr);
        glCompileShader(shader);
        GLint ok = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            GLint len = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
            std::string log(len, '\0');
            glGetShaderInfoLog(shader, len, nullptr, log.data());
            std::fprintf(stderr, "FSR1 compile error (%s): %s\n", filename, log.c_str());
            glDeleteShader(shader);
            return 0;
        }
        GLuint prog = glCreateProgram();
        glAttachShader(prog, shader);
        glLinkProgram(prog);
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if (!ok) {
            GLint len = 0;
            glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
            std::string log(len, '\0');
            glGetProgramInfoLog(prog, len, nullptr, log.data());
            std::fprintf(stderr, "FSR1 link error (%s): %s\n", filename, log.c_str());
            glDeleteProgram(prog);
            prog = 0;
        }
        glDeleteShader(shader);
        return prog;
    };

    GLuint fsr_easu_prog = compile_fsr_program("fsr_easu.glsl");
    GLuint fsr_rcas_prog = compile_fsr_program("fsr_rcas.glsl");

    // FBO: low-res scene color + depth + full-res output + RCAS scratch
    int fb_w = 0, fb_h = 0;
    int sr_w = 0, sr_h = 0;
    gl::Texture fb_color(gl::TextureType::tex_2d);
    gl::Texture fb_depth(gl::TextureType::tex_2d);
    gl::Framebuffer fbo;
    gl::Texture fsr_output(gl::TextureType::tex_2d);
    gl::Texture rcas_scratch(gl::TextureType::tex_2d);
    gl::Framebuffer fsr_blit_fbo(gl::FramebufferType::read);
    gfx::DepthPyramid* depth_pyramid = nullptr;

    auto rebuild_fbo = [&](int full_w, int full_h) {
        sr_w = full_w; sr_h = full_h;
        int lr_w = std::max(1, int(full_w * render_scale));
        int lr_h = std::max(1, int(full_h * render_scale));
        fb_w = lr_w; fb_h = lr_h;

        fb_color = gl::Texture(gl::TextureType::tex_2d);
        fb_color.bind();
        fb_color.image_2d(0, GL_RGBA16F, lr_w, lr_h, GL_RGBA, GL_HALF_FLOAT, nullptr);
        fb_color.parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        fb_color.parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        fb_depth = gl::Texture(gl::TextureType::tex_2d);
        fb_depth.bind();
        fb_depth.image_2d(0, GL_DEPTH_COMPONENT32F, lr_w, lr_h, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        fb_depth.parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        fb_depth.parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        fb_depth.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        fb_depth.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        fbo = gl::Framebuffer();
        fbo.attach_texture(GL_COLOR_ATTACHMENT0, fb_color);
        fbo.attach_texture(GL_DEPTH_ATTACHMENT, fb_depth);
        if (!fbo.check()) {
            std::fprintf(stderr, "FBO incomplete\n");
            std::exit(EXIT_FAILURE);
        }

        // Output texture (EASU writes to this)
        fsr_output = gl::Texture(gl::TextureType::tex_2d);
        fsr_output.bind();
        fsr_output.image_2d(0, GL_RGBA16F, full_w, full_h, GL_RGBA, GL_HALF_FLOAT, nullptr);
        fsr_output.parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        fsr_output.parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // RCAS scratch (RCAS reads fsr_output, writes to rcas_scratch)
        rcas_scratch = gl::Texture(gl::TextureType::tex_2d);
        rcas_scratch.bind();
        rcas_scratch.image_2d(0, GL_RGBA16F, full_w, full_h, GL_RGBA, GL_HALF_FLOAT, nullptr);
        rcas_scratch.parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        rcas_scratch.parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        fsr_blit_fbo = gl::Framebuffer(gl::FramebufferType::read);
        fsr_blit_fbo.attach_texture(GL_COLOR_ATTACHMENT0, rcas_scratch);

        delete depth_pyramid;
        depth_pyramid = new gfx::DepthPyramid(lr_w, lr_h);
    };
    rebuild_fbo(window.framebuffer_width(), window.framebuffer_height());

    float proj_00 = 0.0f, proj_11 = 0.0f;
    gl::enable(GL_DEPTH_TEST);
    gl::clear_color(0.08f, 0.08f, 0.12f, 1.0f);

    gfx::ImGuiOverlay gui;
    gui.init(window);

    bool readback_visible = false;
    gl::Query hiz_timer(gl::QueryType::time_elapsed);

    {
        const glm::mat4& p = cam.projection();
        proj_00 = p[0][0];
        proj_11 = p[1][1];
    }

    gfx::Renderer renderer;
    double last_time = window.time();
    int frame_count = 0;
    int last_fps = 0;
    float prev_render_scale = render_scale;
    GLuint64 last_hiz_time_ns = 0;
    glm::mat4 prev_view_proj(1.0f);

    while (!window.should_close()) {
        float now = window.time();
        window.poll_events();

        cam.set_aspect(float(window.width()) / window.height());

        if (window.key_down(gfx::Key::r)) {
            if (!readback_visible) {
                readback_visible = true;
                gpu.set_readback_active(!gpu.readback_active());
                std::printf("readback %s\n", gpu.readback_active() ? "ON" : "OFF");
            }
        } else {
            readback_visible = false;
        }

        if (hot_prog.poll()) {
            mat.set_program(hot_prog.take_program());
            std::printf("shaders reloaded\n");
        }

        int cur_w = window.framebuffer_width();
        int cur_h = window.framebuffer_height();
        if (cur_w != sr_w || cur_h != sr_h || render_scale != prev_render_scale) {
            prev_render_scale = render_scale;
            rebuild_fbo(cur_w, cur_h);
            const glm::mat4& p = cam.projection();
            proj_00 = p[0][0];
            proj_11 = p[1][1];
        }

        float orbit = now * 0.15f;
        glm::vec3 eye(cos(orbit) * 25.0f, 12.0f, sin(orbit) * 25.0f);
        cam.look_at(eye, {0, 0, 0});

        if (hiz_timer.result_available())
            last_hiz_time_ns = hiz_timer.result();

        hiz_timer.begin();
        depth_pyramid->build(fb_depth.handle());
        hiz_timer.end();
        gpu.set_hiz(depth_pyramid->hiz_texture(), proj_00, proj_11,
                     fb_w, fb_h, depth_pyramid->mip_levels() - 1);

        // Submit all instances
        for (auto& inst : all_instances)
            gpu.draw(inst.handle, inst.transform);

        // --- Scene render at low resolution ---
        fbo.bind();
        renderer.viewport(0, 0, fb_w, fb_h);
        GLenum draw_bufs[] = {GL_COLOR_ATTACHMENT0};
        glDrawBuffers(1, draw_bufs);
        renderer.clear();
        glClear(GL_DEPTH_BUFFER_BIT);

        gfx::Camera render_cam = cam;
        glm::mat4 cur_vp = render_cam.view_projection();

        mat.set_uniform("u_view_proj", cur_vp);
        mat.set_uniform("u_prev_view_proj", prev_view_proj);
        mat.set_uniform("u_render_size", glm::vec2(fb_w, fb_h));
        mat.set_uniform("u_max_instances", static_cast<unsigned int>(gpu.max_instances()));
        mat.bind();

        gpu.flush(render_cam, mat.program_handle());

        prev_view_proj = cur_vp;

        // --- FSR1 upscale (EASU) ---
        gl::Framebuffer::unbind(gl::FramebufferType::both);

        if (fsr_easu_prog && fsr_rcas_prog) {
            // EASU constants
            uint32_t easu_con0[4], easu_con1[4], easu_con2[4], easu_con3[4];
            gfx::fsr_easu_con(easu_con0, (float)sr_w, (float)sr_h, (float)sr_w, (float)sr_h,
                              (float)sr_w, (float)sr_h);
            gfx::fsr_easu_con1(easu_con1, (float)sr_w, (float)sr_h);
            gfx::fsr_easu_con2(easu_con2, (float)sr_w, (float)sr_h);
            gfx::fsr_easu_con3(easu_con3, (float)sr_w, (float)sr_h);
            // Adjust for actual input/output sizes
            {
                float rcp = 1.0f / (float)sr_w;
                reinterpret_cast<float&>(easu_con0[0]) = (float)sr_w * rcp;
                reinterpret_cast<float&>(easu_con0[2]) = 0.5f * (float)sr_w * rcp - 0.5f;
            }
            {
                float rcp = 1.0f / (float)sr_h;
                reinterpret_cast<float&>(easu_con0[1]) = (float)sr_h * rcp;
                reinterpret_cast<float&>(easu_con0[3]) = 0.5f * (float)sr_h * rcp - 0.5f;
            }

            glUseProgram(fsr_easu_prog);
            glUniform4uiv(0, 1, easu_con0);
            glUniform4uiv(1, 1, easu_con1);
            glUniform4uiv(2, 1, easu_con2);
            glUniform4uiv(3, 1, easu_con3);

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, fb_color.handle());
            glBindImageTexture(1, fsr_output.handle(), 0, GL_FALSE, 0,
                               GL_WRITE_ONLY, GL_RGBA16F);

            int groups_x = (sr_w + 7) / 8;
            int groups_y = (sr_h + 7) / 8;
            glDispatchCompute(groups_x, groups_y, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

            // RCAS sharpening
            uint32_t rcas_con[4];
            gfx::fsr_rcas_con(rcas_con, rcas_sharpness);

            glUseProgram(fsr_rcas_prog);
            glUniform4uiv(0, 1, rcas_con);

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, fsr_output.handle());
            glBindImageTexture(1, rcas_scratch.handle(), 0, GL_FALSE, 0,
                               GL_WRITE_ONLY, GL_RGBA16F);

            glDispatchCompute(groups_x, groups_y, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        }

        // --- Blit FSR output to screen ---
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        fsr_blit_fbo.blit_to(0, 0, 0, sr_w, sr_h, 0, 0, sr_w, sr_h,
                              GL_COLOR_BUFFER_BIT, GL_LINEAR);

        gui.begin_frame();
        {
            ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
            ImGui::Begin("GPU Pipeline", nullptr, ImGuiWindowFlags_NoSavedSettings);

            ImGui::SeparatorText("Quality");
            if (ImGui::Combo("Preset", &quality_tier, quality_names, 4)) {
                render_scale = quality_scales[quality_tier];
                rcas_sharpness = quality_sharpness[quality_tier];
            }
            ImGui::SliderFloat("Render Scale", &render_scale, 0.25f, 1.5f, "%.2f");
            ImGui::SliderFloat("Sharpness", &rcas_sharpness, 0.0f, 1.0f, "%.2f");

            ImGui::SeparatorText("Dynamic Resolution");
            ImGui::Checkbox("Auto DRS", &drs_enabled);
            ImGui::SliderFloat("Target ms", &drs_target_ms, 1.0f, 16.0f, "%.1f");

            ImGui::SeparatorText("Stats");
            ImGui::Text("Instances: %d / %zu", gpu.last_visible_count(), all_instances.size());
            ImGui::Text("HiZ: %llu us", (unsigned long long)(last_hiz_time_ns / 1000));
            ImGui::Text("Cull: %llu us", (unsigned long long)gpu.last_cull_time_us());
            ImGui::Text("Pack: %llu us", (unsigned long long)gpu.last_pack_time_us());
            ImGui::Text("Draw: %llu us", (unsigned long long)gpu.last_draw_time_us());
            if (last_fps > 0)
                ImGui::Text("FPS: %d", last_fps);
            if (drs_enabled)
                ImGui::Text("DRS: %.2f ms (target %.1f)", drs_smoothed_ms, drs_target_ms);
            ImGui::End();
        }

        gui.render();

        window.swap_buffers();

        // DRS: adjust render_scale based on frame time
        if (drs_enabled && last_fps > 0) {
            static double drs_last = 0.0;
            double frame_ms = (now - drs_last) * 1000.0;
            drs_last = now;
            if (drs_last != 0.0) {
                float alpha = 0.1f;
                drs_smoothed_ms = drs_smoothed_ms * (1.0f - alpha) + static_cast<float>(frame_ms) * alpha;
                float error = drs_smoothed_ms - drs_target_ms;
                float adjustment = 1.0f + error * -0.05f;
                float new_scale = render_scale * adjustment;
                new_scale = std::clamp(new_scale, 0.25f, 1.5f);
                new_scale = std::round(new_scale * 100.0f) / 100.0f;
                if (std::abs(new_scale - render_scale) >= 0.01f) {
                    render_scale = new_scale;
                }
            }
        }

        ++frame_count;
        if (now - last_time >= 1.0) {
            last_fps = frame_count;
            gpu.debug_readback();
            int tris = 0;
            for (int s = 0; s < gpu.slot_count(); ++s)
                tris += slot_tris[s] * gpu.last_lod_visible(s);
            std::printf("FPS: %d | visible: %d / %zu | tris: %d"
                        " | HiZ: %llu us | cull: %llu us | pack: %llu us | draw: %llu us\n",
                        last_fps, gpu.last_visible_count(), all_instances.size(), tris,
                        (unsigned long long)(last_hiz_time_ns / 1000),
                        (unsigned long long)gpu.last_cull_time_us(),
                        (unsigned long long)gpu.last_pack_time_us(),
                        (unsigned long long)gpu.last_draw_time_us());
            frame_count = 0;
            last_time = now;
        }
    }

    glDeleteProgram(fsr_easu_prog);
    glDeleteProgram(fsr_rcas_prog);

    return EXIT_SUCCESS;
}
