// Example 26 — Deferred rendering: geometry pass → G-buffer → lighting pass.
// Users can replace the lighting shader to create their own pipeline.

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>
#include <gfx/gbuffer.hpp>
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
// Geometry pass — writes albedo, normal/roughness/metallic, emissive/AO, depth
// ---------------------------------------------------------------------------

static const char* geo_vs = R"(
#version 430 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;

uniform mat4 u_model;
uniform mat4 u_view_proj;
uniform mat3 u_normal_mat;

out vec3 v_pos;
out vec3 v_normal;
out vec2 v_uv;

void main() {
    vec4 world = u_model * vec4(a_pos, 1.0);
    gl_Position = u_view_proj * world;
    v_pos = world.xyz;
    v_normal = normalize(u_normal_mat * a_normal);
    v_uv = a_uv;
}
)";

static const char* geo_fs = R"(
#version 430 core
in vec3 v_pos;
in vec3 v_normal;
in vec2 v_uv;

layout(location = 0) out vec4 out_albedo;
layout(location = 1) out vec4 out_normal_rm;
layout(location = 2) out vec4 out_emissive_ao;

uniform vec3 u_albedo_factor;
uniform float u_metallic;
uniform float u_roughness;
uniform float u_ao;

uniform bool u_has_base_tex;
uniform bool u_has_mr_tex;
uniform bool u_has_occlusion_tex;

uniform sampler2D u_base_tex;
uniform sampler2D u_mr_tex;
uniform sampler2D u_occlusion_tex;

void main() {
    vec4 base = u_has_base_tex ? texture(u_base_tex, v_uv) : vec4(1.0);
    base.rgb *= u_albedo_factor;

    vec2 mr = u_has_mr_tex ? texture(u_mr_tex, v_uv).bg : vec2(1.0);
    float metal = mr.x * u_metallic;
    float rough = max(mr.y * u_roughness, 0.001);

    float ao = u_has_occlusion_tex ? texture(u_occlusion_tex, v_uv).r * u_ao : u_ao;

    vec3 N = normalize(v_normal);

    out_albedo = vec4(base.rgb, 1.0);
    out_normal_rm = vec4(N.xy * 0.5 + 0.5, rough, metal);
    out_emissive_ao = vec4(0.0, 0.0, 0.0, ao);
}
)";

// ---------------------------------------------------------------------------
// Lighting pass — fullscreen quad, reads G-buffer, computes PBR + IBL + shadow
// ---------------------------------------------------------------------------

static const char* light_vs = R"(
#version 430 core
layout(location = 0) in vec2 a_pos;
out vec2 v_uv;
void main() {
    v_uv = a_pos * 0.5 + 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
)";

static const char* light_fs = R"(
#version 430 core
in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

uniform sampler2D u_albedo_tex;
uniform sampler2D u_normal_rm_tex;
uniform sampler2D u_emissive_ao_tex;
uniform sampler2D u_depth_tex;

uniform mat4 u_inv_view_proj;
uniform vec3 u_view_pos;
uniform vec3 u_light_dir;

uniform samplerCube u_irradiance_map;
uniform samplerCube u_prefilter_map;
uniform sampler2D u_brdf_lut;
uniform int u_prefilter_levels;

uniform sampler2D u_shadow_map;
uniform mat4 u_light_vp;
uniform bool u_has_shadow;

uniform samplerCube u_skybox_map;

uniform vec3 u_ambient_top;
uniform vec3 u_ambient_bottom;
uniform float u_ambient_intensity;

vec3 aces(vec3 x) {
    float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 fresnel_schlick_roughness(float cos_theta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(max(1.0 - cos_theta, 0.0), 5.0);
}

float shadow_factor(vec4 frag_light_space, sampler2D shadow_map) {
    vec3 proj = frag_light_space.xyz / frag_light_space.w;
    proj = proj * 0.5 + 0.5;
    if (proj.x < 0 || proj.x > 1 || proj.y < 0 || proj.y > 1 || proj.z < 0 || proj.z > 1)
        return 1.0;
    float bias = 0.001;
    return (proj.z - bias) > texture(shadow_map, proj.xy).r ? 0.0 : 1.0;
}

void main() {
    ivec2 tex_size = textureSize(u_depth_tex, 0);
    vec2 uv = gl_FragCoord.xy / vec2(tex_size);

    float depth = texture(u_depth_tex, uv).r;
    vec4 clip = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 world = u_inv_view_proj * clip;
    vec3 pos = world.xyz / world.w;

    if (depth >= 1.0) {
        vec3 dir = normalize(pos - u_view_pos);
        vec3 sky = texture(u_skybox_map, dir).rgb;
        frag_color = vec4(pow(sky, vec3(1.0 / 2.2)), 1.0);
        return;
    }

    vec3 albedo = texture(u_albedo_tex, uv).rgb;

    vec4 nrm = texture(u_normal_rm_tex, uv);
    vec3 N;
    N.xy = nrm.xy * 2.0 - 1.0;
    N.z = sqrt(max(1.0 - dot(N.xy, N.xy), 0.0));
    float roughness = nrm.b;
    float metallic = nrm.a;

    float ao = texture(u_emissive_ao_tex, uv).a;

    vec3 V = normalize(u_view_pos - pos);
    float NdotV = max(dot(N, V), 0.0);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 kS = fresnel_schlick_roughness(NdotV, F0, roughness);
    vec3 kD = (1.0 - kS) * (1.0 - metallic);

    vec3 irradiance = texture(u_irradiance_map, N).rgb;
    vec3 diffuse = kD * irradiance * albedo;

    vec3 R = reflect(-V, N);
    float level = roughness * float(u_prefilter_levels - 1);
    vec3 prefilter = textureLod(u_prefilter_map, R, level).rgb;
    vec2 brdf = texture(u_brdf_lut, vec2(NdotV, roughness)).rg;
    vec3 specular = prefilter * (kS * brdf.x + brdf.y);

    vec3 ambient = (diffuse + specular) * ao;

    vec3 hemi = mix(u_ambient_bottom, u_ambient_top, N.y * 0.5 + 0.5);
    ambient += hemi * u_ambient_intensity;

    // Diagnostic: output irradiance directly to check IBL
    frag_color = vec4(irradiance, 1.0);
    return;

    float shadow = 1.0;
    if (u_has_shadow) {
        vec4 light_pos = u_light_vp * vec4(pos, 1.0);
        shadow = shadow_factor(light_pos, u_shadow_map);
    }
    ambient *= mix(0.15, 1.0, shadow);

    vec3 color = aces(ambient);
    color = pow(color, vec3(1.0 / 2.2));

    frag_color = vec4(color, 1.0);
}
)";

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

    glm::vec3 new_pos = cam.position() + vel;
    glm::vec3 dir(std::cos(pitch) * sy, std::sin(pitch), std::cos(pitch) * cy);
    cam.look_at(new_pos, new_pos + dir);
}

// ---------------------------------------------------------------------------
// Shader compilation helper
// ---------------------------------------------------------------------------

static gl::Program* make_program(const char* vs_src, const char* fs_src) {
    gl::Shader vs(gl::ShaderType::vertex, vs_src);
    gl::Shader fs(gl::ShaderType::fragment, fs_src);
    if (!vs.compiled() || !fs.compiled()) return nullptr;
    auto* prog = new gl::Program;
    prog->attach(vs);
    prog->attach(fs);
    if (!prog->link()) { delete prog; return nullptr; }
    return prog;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    gllib::log_to_stderr(gllib::LogLevel::info);
    gfx::Window window({"26 Deferred PBR - Sponza", 1200, 800});

    // --- IBL ---
    gfx::IBLProbe ibl;
    ibl.generate_procedural(256);
    ibl.bake();

    // --- Skybox ---
    gfx::Skybox skybox(ibl.env_map());

    // --- Shadow map ---
    gfx::ShadowMap shadow(8192);

    // --- ImGui ---
    gfx::ImGuiOverlay gui;
    if (!gui.init(window)) {
        gllib::log(gllib::LogLevel::error, "ImGui init failed");
        return EXIT_FAILURE;
    }

    // --- Shaders ---
    gl::Program* geo_prog = make_program(geo_vs, geo_fs);
    gl::Program* light_prog = make_program(light_vs, light_fs);
    if (!geo_prog || !light_prog) {
        gllib::log(gllib::LogLevel::error, "Shader compilation failed");
        return EXIT_FAILURE;
    }

    // --- Fullscreen quad ---
    const float quad_verts[] = {-1, -1, 1, -1, 1, 1, -1, 1};
    const unsigned int quad_idx[] = {0, 1, 2, 0, 2, 3};
    gl::Buffer quad_vbo(gl::BufferType::vertex);
    quad_vbo.data(quad_verts, sizeof(quad_verts));
    gl::Buffer quad_ebo(gl::BufferType::index);
    quad_ebo.data(quad_idx, sizeof(quad_idx));
    gl::VertexArray quad_vao;
    quad_vao.bind();
    quad_vbo.bind();
    quad_ebo.bind();
    quad_vao.attrib_pointer(0, 2, GL_FLOAT, false, 8, (void*)0);
    quad_vao.enable_attrib(0);
    gl::VertexArray::unbind();

    // --- GBuffer ---
    gfx::GBuffer gbuf;
    gbuf.create(window.width(), window.height());

    // --- Load model ---
    gfx::Model model;
    if (!model.load("sponza.glb")) {
        gllib::log(gllib::LogLevel::error, "Failed to load Sponza model");
        return EXIT_FAILURE;
    }
    gllib::logf(gllib::LogLevel::info,
                "Loaded Sponza: %zu meshes, %zu materials, %zu textures",
                model.mesh_count(), model.material_count(), model.texture_count());

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
    renderer.set_clear_color(0.0f, 0.0f, 0.0f, 1.0f);
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

        // Resize GBuffer on window resize
        if (window.width() != gbuf.width() || window.height() != gbuf.height()) {
            gbuf.create(window.width(), window.height());
        }

        glm::mat4 vp = cam.view_projection();
        glm::mat4 inv_vp = glm::inverse(vp);

        glm::mat4 light_vp = gfx::compute_light_vp(
            vp, light_dir, scene_radius, 10.0f);

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

        // --- Geometry pass ---
        gbuf.bind_for_geometry();

        geo_prog->use();
        auto geo_loc = [&](const char* n) { return geo_prog->uniform_location(n); };
        GLint loc;

        loc = geo_loc("u_view_proj"); if (loc >= 0) geo_prog->uniform_matrix4fv(loc, glm::value_ptr(vp));

        for (size_t idx : opaque) {
            int mi = model.mesh_material(int(idx));
            auto& mat = model.material_info(size_t(mi >= 0 ? mi : 0));

            loc = geo_loc("u_model"); if (loc >= 0) geo_prog->uniform_matrix4fv(loc, glm::value_ptr(model_mat));
            glm::mat3 nm = glm::transpose(glm::inverse(glm::mat3(model_mat)));
            loc = geo_loc("u_normal_mat"); if (loc >= 0) geo_prog->uniform_matrix3fv(loc, glm::value_ptr(nm));

            loc = geo_loc("u_albedo_factor"); if (loc >= 0) geo_prog->uniform3fv(loc, mat.base_color_factor);
            loc = geo_loc("u_metallic"); if (loc >= 0) geo_prog->uniform1f(loc, mat.metallic_factor);
            loc = geo_loc("u_roughness"); if (loc >= 0) geo_prog->uniform1f(loc, mat.roughness_factor);
            loc = geo_loc("u_ao"); if (loc >= 0) geo_prog->uniform1f(loc, 1.0f);

            if (mat.base_color_tex >= 0 && size_t(mat.base_color_tex) < model.texture_count()) {
                model.texture(mat.base_color_tex)->bind(0);
                loc = geo_loc("u_base_tex"); if (loc >= 0) geo_prog->uniform1i(loc, 0);
                loc = geo_loc("u_has_base_tex"); if (loc >= 0) geo_prog->uniform1i(loc, 1);
            } else {
                loc = geo_loc("u_has_base_tex"); if (loc >= 0) geo_prog->uniform1i(loc, 0);
            }

            if (mat.metallic_roughness_tex >= 0 && size_t(mat.metallic_roughness_tex) < model.texture_count()) {
                model.texture(mat.metallic_roughness_tex)->bind(1);
                loc = geo_loc("u_mr_tex"); if (loc >= 0) geo_prog->uniform1i(loc, 1);
                loc = geo_loc("u_has_mr_tex"); if (loc >= 0) geo_prog->uniform1i(loc, 1);
            } else {
                loc = geo_loc("u_has_mr_tex"); if (loc >= 0) geo_prog->uniform1i(loc, 0);
            }

            if (mat.occlusion_tex >= 0 && size_t(mat.occlusion_tex) < model.texture_count()) {
                model.texture(mat.occlusion_tex)->bind(2);
                loc = geo_loc("u_occlusion_tex"); if (loc >= 0) geo_prog->uniform1i(loc, 2);
                loc = geo_loc("u_has_occlusion_tex"); if (loc >= 0) geo_prog->uniform1i(loc, 1);
            } else {
                loc = geo_loc("u_has_occlusion_tex"); if (loc >= 0) geo_prog->uniform1i(loc, 0);
            }

            model.mesh(idx).draw();
        }

        gl::disable(GL_DEPTH_TEST);

        // --- Lighting pass (includes skybox for background pixels) ---
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, window.width(), window.height());
        gl::clear_color(0.08f, 0.08f, 0.10f, 1.0f);
        gl::clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        gbuf.bind_for_lighting(3, 4, 5, 6);

        light_prog->use();
        auto light_loc = [&](const char* n) { return light_prog->uniform_location(n); };

        loc = light_loc("u_albedo_tex");      if (loc >= 0) light_prog->uniform1i(loc, 3);
        loc = light_loc("u_normal_rm_tex");    if (loc >= 0) light_prog->uniform1i(loc, 4);
        loc = light_loc("u_emissive_ao_tex");  if (loc >= 0) light_prog->uniform1i(loc, 5);
        loc = light_loc("u_depth_tex");        if (loc >= 0) light_prog->uniform1i(loc, 6);
        loc = light_loc("u_inv_view_proj");    if (loc >= 0) light_prog->uniform_matrix4fv(loc, glm::value_ptr(inv_vp));
        loc = light_loc("u_view_pos");         if (loc >= 0) light_prog->uniform3fv(loc, glm::value_ptr(cam.position()));
        loc = light_loc("u_light_dir");        if (loc >= 0) light_prog->uniform3fv(loc, glm::value_ptr(light_dir));

        ibl.env_map().bind(7);
        loc = light_loc("u_skybox_map");       if (loc >= 0) light_prog->uniform1i(loc, 7);

        ibl.irradiance_map().bind(8);
        ibl.prefilter_map().bind(9);
        ibl.brdf_lut().bind(10);
        loc = light_loc("u_irradiance_map");   if (loc >= 0) light_prog->uniform1i(loc, 8);
        loc = light_loc("u_prefilter_map");    if (loc >= 0) light_prog->uniform1i(loc, 9);
        loc = light_loc("u_brdf_lut");         if (loc >= 0) light_prog->uniform1i(loc, 10);
        loc = light_loc("u_prefilter_levels"); if (loc >= 0) light_prog->uniform1i(loc, ibl.prefilter_map().levels());

        shadow.bind(11);
        loc = light_loc("u_shadow_map");       if (loc >= 0) light_prog->uniform1i(loc, 11);
        loc = light_loc("u_light_vp");         if (loc >= 0) light_prog->uniform_matrix4fv(loc, glm::value_ptr(light_vp));
        loc = light_loc("u_has_shadow");       if (loc >= 0) light_prog->uniform1i(loc, 1);

        loc = light_loc("u_ambient_top");      if (loc >= 0) light_prog->uniform3f(loc, 0.3f, 0.4f, 0.6f);
        loc = light_loc("u_ambient_bottom");   if (loc >= 0) light_prog->uniform3f(loc, 0.1f, 0.08f, 0.06f);
        loc = light_loc("u_ambient_intensity"); if (loc >= 0) light_prog->uniform1f(loc, 0.15f);

        quad_vao.bind();
        gl::draw_elements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

        // --- ImGui ---
        gui.begin_frame();
        {
            ImGui::Begin("G-Buffer");
            float sz = float(ImGui::GetContentRegionAvail().x);
            if (sz < 64) sz = 256;
            ImGui::Image((ImTextureID)(intptr_t)gbuf.albedo_handle(), ImVec2(sz, sz * 0.5f), ImVec2(0,1), ImVec2(1,0));
            ImGui::Image((ImTextureID)(intptr_t)gbuf.normal_rm_handle(), ImVec2(sz, sz * 0.5f), ImVec2(0,1), ImVec2(1,0));
            ImGui::Image((ImTextureID)(intptr_t)gbuf.emissive_ao_handle(), ImVec2(sz, sz * 0.5f), ImVec2(0,1), ImVec2(1,0));
            ImGui::Image((ImTextureID)(intptr_t)gbuf.depth_handle(), ImVec2(sz, sz * 0.5f), ImVec2(0,1), ImVec2(1,0));
            ImGui::Text("Shadow: %dx%d", shadow.size(), shadow.size());
            ImGui::End();
        }
        gui.render();

        window.swap_buffers();
        window.poll_events();
    }

    delete geo_prog;
    delete light_prog;

    return EXIT_SUCCESS;
}
