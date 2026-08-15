// Example 35 — SSRC: Screen Space Radiance Cascades (radiance cascades GI).
//
// The scene is a classic Cornell box whose only light is the emissive ceiling
// rectangle. Since an emissive surface is representable as geometry, there is
// no analytic direct-lighting pass at all: the cascades resolve the area
// light's direct contribution and all bounces in a single trace. (Delta lights
// — point/directional — can't be intersected by marching and would need a
// separate forward pass, but this example has none.)
//
// A hierarchical radiance field is built in screen space: the G-buffer's
// surface radiance (here just emission) is the finest "source" field, and
// coarser cascades ray-march a fixed number of directions over progressively
// larger distance intervals, falling back (merging) to the next coarser
// cascade on a miss. A gather pass reconstructs diffuse irradiance at every
// pixel by summing each cascade's stored radiance over its directions,
// cosine-weighted over the upper hemisphere.
//
// Layout per cascade: directions are packed along X, probes along Y.
//   texel (d * gridW + px, py)  =  radiance at probe (px,py) for direction d.
// Level 0 is not stored: it is the source radiance field sampled directly.
//
// Pipeline:
//   geometry pass  -> 6 MRT G-buffer at "ssrc" resolution
//   cascade build  -> coarsest-to-finest compute, one dispatch per cascade
//   gather pass    -> compute, integrates all cascades into a HDR target
//   display pass   -> fullscreen triangle, exposure + tonemap + gamma

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>
#include <gllib/log.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {
constexpr int MAX_CASCADES = 10;

// ===========================================================================
// Shaders
// ===========================================================================

const char* gbuf_vs = R"(
#version 460 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;

uniform mat4 u_model;
uniform mat4 u_view_proj;

out vec3 v_pos;
out vec3 v_normal;

void main() {
    vec4 world = u_model * vec4(a_pos, 1.0);
    gl_Position = u_view_proj * world;
    v_pos = world.xyz;
    v_normal = normalize(mat3(u_model) * a_normal);
}
)";

const char* gbuf_fs = R"(
#version 460 core
in vec3 v_pos;
in vec3 v_normal;

layout(location = 0) out vec4 out_albedo;
layout(location = 1) out vec4 out_normal;
layout(location = 2) out vec4 out_position;
layout(location = 3) out vec4 out_surface;
layout(location = 4) out vec4 out_emissive;
layout(location = 5) out float out_depth;

uniform vec3 u_albedo;
uniform vec3 u_emissive;
uniform float u_emissive_scale;
uniform mat4 u_view;

vec2 octahedral(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 p = n.xy;
    if (n.z < 0.0) {
        p = (1.0 - abs(p.yx)) * mix(vec2(-1.0), vec2(1.0), step(0.0, p));
    }
    return p * 0.5 + 0.5;
}

void main() {
    vec3 N = normalize(v_normal);
    vec3 albedo = u_albedo;
    vec3 emissive = u_emissive * u_emissive_scale;

    // There is no point/directional light in this example: the only light
    // source is the emissive ceiling rectangle, which is representable as
    // geometry. The "radiance leaving this surface" buffer is therefore just
    // emission here; the cascades resolve direct AND indirect light from the
    // area light in a single trace (delta lights can't be hit by marching and
    // would need a separate analytic pass).
    vec3 radiance = emissive;

    out_albedo   = vec4(albedo, 1.0);
    out_normal   = vec4(octahedral(N), 0.0, 1.0);
    out_position = vec4(v_pos, 1.0);
    out_surface  = vec4(radiance, 1.0);
    out_emissive = vec4(emissive, 1.0);

    vec4 view_p = u_view * vec4(v_pos, 1.0);
    out_depth = -view_p.z;
}
)";

const char* build_cs = R"(
#version 460 core
layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0) uniform sampler2D u_pos_tex;
layout(binding = 1) uniform sampler2D u_normal_tex;
layout(binding = 2) uniform sampler2D u_surface_tex;
layout(r32f, binding = 3) uniform readonly image2D u_depth_tex;
layout(rgba16f, binding = 4) uniform writeonly image2D u_out;
layout(binding = 5) uniform sampler2D u_next_tex;

uniform mat4 u_view_proj;
uniform mat4 u_view;
uniform int u_ssrc_w;
uniform int u_ssrc_h;
uniform float u_far;

uniform int u_grid_w;        // current cascade probe grid
uniform int u_rays;          // current cascade rays per dimension
uniform float u_spacing;     // current cascade probe spacing (px)
uniform int u_next_rays;     // coarser cascade rays per dimension
uniform float u_next_spacing;// coarser cascade probe spacing (px)
uniform int u_next_grid_w;
uniform int u_next_tex_w;
uniform int u_next_tex_h;
uniform float u_interval_start;
uniform float u_interval_end;
uniform int u_has_next;
uniform vec3 u_ambient_color;
uniform float u_ambient_intensity;
uniform float u_ray_bias;
uniform float u_thickness;
uniform int u_steps;
uniform float u_seed;

const float PI = 3.14159265359;

vec3 oct_decode(vec2 e) {
    e = e * 2.0 - 1.0;
    vec3 n = vec3(e, 1.0);
    n.z = 1.0 - abs(n.x) - abs(n.y);
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(n.yx)) * mix(vec2(-1.0), vec2(1.0), step(0.0, n.xy));
    }
    return normalize(n);
}

float ig_noise(vec2 p, float seed) {
    vec2 q = p + seed * vec2(0.123, 0.456);
    return fract(52.9829189 * fract(0.06711056 * q.x + 0.00583715 * q.y));
}

vec2 concentric_disk(vec2 p) {
    float a = 2.0 * p.x - 1.0;
    float b = 2.0 * p.y - 1.0;
    float r, phi;
    if (a > -b) {
        if (a > b)      { r = a; phi = (PI / 4.0) * (b / max(a, 1e-6)); }
        else            { r = b; phi = (PI / 4.0) * (2.0 - a / max(b, 1e-6)); }
    } else {
        if (a < b)      { r = -a; phi = (PI / 4.0) * (4.0 + b / max(a, 1e-6)); }
        else            { r = -b; phi = (PI / 4.0) * (6.0 - a / max(b, 1e-6)); }
    }
    return r * vec2(cos(phi), sin(phi));
}

vec3 dir_world(int d, vec3 t, vec3 b, vec3 n) {
    int r = u_rays;
    int dx = d % r;
    int dy = d / r;
    vec2 disk = concentric_disk((vec2(dx, dy) + 0.5) / float(r));
    vec3 dir = t * disk.x + b * disk.y + n * sqrt(max(1.0 - dot(disk, disk), 0.0));
    return normalize(dir);
}

// Returns true on a surface hit within [t0, t1]; rad is the surface radiance.
bool march(vec3 origin, vec3 dir, float t0, float t1, ivec2 px, float seed, out vec3 rad) {
    int steps = u_steps;
    float t = t0 + (t1 - t0) * ig_noise(vec2(px), seed);
    float dt = (t1 - t0) / float(steps);
    for (int i = 0; i < steps; ++i) {
        vec3 pos = origin + dir * t;
        vec4 clip = u_view_proj * vec4(pos, 1.0);
        if (clip.w > 1e-4) {
            vec2 ndc = clip.xy / clip.w;
            if (ndc.x >= -1.0 && ndc.x <= 1.0 && ndc.y >= -1.0 && ndc.y <= 1.0) {
                vec2 uv = ndc * 0.5 + 0.5;
                ivec2 spx = ivec2(uv * vec2(u_ssrc_w, u_ssrc_h));
                float surf_depth = imageLoad(u_depth_tex, spx).r;
                vec4 view_p = u_view * vec4(pos, 1.0);
                float ray_depth = -view_p.z;
                if (ray_depth > surf_depth + u_thickness) {
                    rad = texture(u_surface_tex, uv).rgb;
                    return true;
                }
            }
        }
        t += dt;
    }
    return false;
}

vec3 merge_value(int d, int px, int py) {
    float d_next = (float(d) + 0.5) *
        float(u_next_rays * u_next_rays) / float(u_rays * u_rays) - 0.5;
    float px_c = (float(px) + 0.5) * u_spacing / u_next_spacing;
    float py_c = (float(py) + 0.5) * u_spacing / u_next_spacing;
    float xtex = d_next * float(u_next_grid_w) + px_c * float(u_next_grid_w);
    float ytex = py_c;
    vec2 muv = (vec2(xtex, ytex) + 0.5) / vec2(float(u_next_tex_w), float(u_next_tex_h));
    return texture(u_next_tex, muv).rgb;
}

void main() {
    ivec2 t = ivec2(gl_GlobalInvocationID.xy);
    int d = t.x / u_grid_w;
    int px = t.x % u_grid_w;
    int py = t.y;

    vec2 uv = (vec2(px, py) + 0.5) * u_spacing / vec2(u_ssrc_w, u_ssrc_h);
    vec4 P = texture(u_pos_tex, uv);
    if (P.w < 0.5) {
        vec3 v = (u_has_next == 1)
            ? merge_value(d, px, py)
            : u_ambient_color * u_ambient_intensity;
        imageStore(u_out, t, vec4(v, 1.0));
        return;
    }

    vec3 origin = P.xyz;
    vec3 n = oct_decode(texture(u_normal_tex, uv).xy);

    vec3 tng = normalize(cross(vec3(0.0, 1.0, 0.0), n));
    if (dot(tng, tng) < 1e-4) tng = normalize(cross(vec3(1.0, 0.0, 0.0), n));
    vec3 bin = cross(n, tng);

    vec3 dir = dir_world(d, tng, bin, n);
    origin += n * u_ray_bias;

    vec3 rad;
    float seed = u_seed + float(d) * 0.137;
    if (march(origin, dir, u_interval_start, u_interval_end, ivec2(px, py), seed, rad)) {
        imageStore(u_out, t, vec4(rad, 1.0));
    } else if (u_has_next == 1) {
        imageStore(u_out, t, vec4(merge_value(d, px, py), 1.0));
    } else {
        imageStore(u_out, t, vec4(u_ambient_color * u_ambient_intensity, 1.0));
    }
}
)";

const char* gather_cs = R"(
#version 460 core
#define MAX_CASCADES 10
layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba16f, binding = 0) uniform readonly image2D u_albedo_tex;
layout(rgba16f, binding = 1) uniform readonly image2D u_normal_tex;
layout(rgba16f, binding = 2) uniform readonly image2D u_emissive_tex;
layout(rgba16f, binding = 3) uniform readonly image2D u_position_tex;
layout(r32f, binding = 4) uniform readonly image2D u_depth_tex;
layout(rgba16f, binding = 5) uniform writeonly image2D u_out;

layout(binding = 10) uniform sampler2D u_surface_tex;
layout(binding = 11) uniform sampler2D u_cascades[MAX_CASCADES];

uniform int u_ssrc_w;
uniform int u_ssrc_h;
uniform float u_far;
uniform int u_num_cascades;
uniform int u_rays[MAX_CASCADES];
uniform int u_spacing[MAX_CASCADES];
uniform int u_grid_w[MAX_CASCADES];
uniform int u_tex_w[MAX_CASCADES];
uniform int u_tex_h[MAX_CASCADES];

uniform float u_gi_strength;
uniform float u_near_strength;
uniform int u_view_mode;
uniform int u_debug_level;

const float PI = 3.14159265359;

vec3 oct_decode(vec2 e) {
    e = e * 2.0 - 1.0;
    vec3 n = vec3(e, 1.0);
    n.z = 1.0 - abs(n.x) - abs(n.y);
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(n.yx)) * mix(vec2(-1.0), vec2(1.0), step(0.0, n.xy));
    }
    return normalize(n);
}

vec2 concentric_disk(vec2 p) {
    float a = 2.0 * p.x - 1.0;
    float b = 2.0 * p.y - 1.0;
    float r, phi;
    if (a > -b) {
        if (a > b)      { r = a; phi = (PI / 4.0) * (b / max(a, 1e-6)); }
        else            { r = b; phi = (PI / 4.0) * (2.0 - a / max(b, 1e-6)); }
    } else {
        if (a < b)      { r = -a; phi = (PI / 4.0) * (4.0 + b / max(a, 1e-6)); }
        else            { r = -b; phi = (PI / 4.0) * (6.0 - a / max(b, 1e-6)); }
    }
    return r * vec2(cos(phi), sin(phi));
}

vec3 dir_world(int d, int r, vec3 t, vec3 b, vec3 n) {
    int dx = d % r;
    int dy = d / r;
    vec2 disk = concentric_disk((vec2(dx, dy) + 0.5) / float(r));
    vec3 dir = t * disk.x + b * disk.y + n * sqrt(max(1.0 - dot(disk, disk), 0.0));
    return normalize(dir);
}

// Bilinear sample of a stored cascade (level >= 1) at this pixel, direction d.
vec3 sample_cascade(int level, ivec2 px, float d) {
    float spacing = float(u_spacing[level]);
    float px_c = (float(px.x) + 0.5) / spacing;
    float py_c = (float(px.y) + 0.5) / spacing;
    float xtex = (d + 0.5) * float(u_grid_w[level]) + px_c * float(u_grid_w[level]);
    float ytex = py_c;
    vec2 uv = (vec2(xtex, ytex) + 0.5) / vec2(float(u_tex_w[level]), float(u_tex_h[level]));
    return texture(u_cascades[level - 1], uv).rgb;
}

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);

    vec3 albedo  = imageLoad(u_albedo_tex, px).rgb;
    vec3 emissive = imageLoad(u_emissive_tex, px).rgb;
    float surf_mark = imageLoad(u_albedo_tex, px).a;
    vec3 n = oct_decode(imageLoad(u_normal_tex, px).xy);

    vec3 tng = normalize(cross(vec3(0.0, 1.0, 0.0), n));
    if (dot(tng, tng) < 1e-4) tng = normalize(cross(vec3(1.0, 0.0, 0.0), n));
    vec3 bin = cross(n, tng);

    if (u_view_mode == 1) { imageStore(u_out, px, vec4(albedo, 1.0)); return; }
    if (u_view_mode == 2) { imageStore(u_out, px, vec4(n * 0.5 + 0.5, 1.0)); return; }
    if (u_view_mode == 3) {
        vec3 p = imageLoad(u_position_tex, px).rgb;
        imageStore(u_out, px, vec4(clamp(p * 0.25 + 0.5, 0.0, 1.0), 1.0));
        return;
    }
    if (u_view_mode == 4) {
        vec2 suv = (vec2(px) + 0.5) / vec2(u_ssrc_w, u_ssrc_h);
        imageStore(u_out, px, vec4(texture(u_surface_tex, suv).rgb, 1.0));
        return;
    }
    if (u_view_mode == 5) { imageStore(u_out, px, vec4(emissive, 1.0)); return; }
    if (u_view_mode == 6) {
        float depth = imageLoad(u_depth_tex, px).r / u_far;
        imageStore(u_out, px, vec4(vec3(depth), 1.0));
        return;
    }
    if (u_view_mode == 7) {
        int lvl = clamp(u_debug_level, 0, u_num_cascades - 1);
        vec3 c;
        if (lvl == 0) {
            vec2 suv = (vec2(px) + 0.5) / vec2(u_ssrc_w, u_ssrc_h);
            c = texture(u_surface_tex, suv).rgb;
        } else {
            c = sample_cascade(lvl, px, 0.0);
        }
        imageStore(u_out, px, vec4(c, 1.0));
        return;
    }

    vec3 E = vec3(0.0);
    for (int c = 0; c < u_num_cascades; ++c) {
        int r = u_rays[c];
        float omega = 4.0 * PI / float(r * r);
        for (int d = 0; d < r * r; ++d) {
            vec3 dir = dir_world(d, r, tng, bin, n);
            float cosw = max(dot(n, dir), 0.0);
            if (cosw <= 0.0) continue;
            vec3 rad;
            if (c == 0) {
                vec2 suv = (vec2(px) + 0.5) / vec2(u_ssrc_w, u_ssrc_h);
                rad = texture(u_surface_tex, suv).rgb * u_near_strength;
            } else {
                rad = sample_cascade(c, px, float(d));
            }
            E += rad * cosw * omega;
        }
    }

    vec3 final = albedo / PI * (E * u_gi_strength) + emissive;
    if (surf_mark < 0.5) final = vec3(0.0);
    imageStore(u_out, px, vec4(final, 1.0));
}
)";

const char* display_vs = R"(
#version 460 core
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";

const char* display_fs = R"(
#version 460 core
in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_color;
uniform float u_exposure;
uniform int u_tonemap;
uniform float u_gamma;

vec3 aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 c = texture(u_color, v_uv).rgb * u_exposure;
    if (u_tonemap == 1) c = aces(c);
    else c = c / (1.0 + c);
    c = pow(max(c, vec3(0.0)), vec3(1.0 / u_gamma));
    frag_color = vec4(c, 1.0);
}
)";

// ===========================================================================
// Host helpers
// ===========================================================================

struct Ssrc {
    int w = 0, h = 0;
    gl::Texture albedo{gl::TextureType::tex_2d};
    gl::Texture normal{gl::TextureType::tex_2d};
    gl::Texture position{gl::TextureType::tex_2d};
    gl::Texture surface{gl::TextureType::tex_2d};
    gl::Texture emissive{gl::TextureType::tex_2d};
    gl::Texture depth{gl::TextureType::tex_2d};
    gl::Texture color{gl::TextureType::tex_2d};
    gl::Renderbuffer depth_rbo;
    gl::Framebuffer fbo;
};

void create_ssrc_textures(Ssrc& s, int w, int h) {
    s.w = w;
    s.h = h;
    auto tex2 = [&](gl::Texture& t) {
        t.image_2d(0, GL_RGBA16F, w, h, GL_RGBA, GL_FLOAT, nullptr, 1);
        t.parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        t.parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        t.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        t.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    };
    tex2(s.albedo);
    tex2(s.normal);
    tex2(s.position);
    tex2(s.surface);
    tex2(s.emissive);
    tex2(s.color);

    s.depth.image_2d(0, GL_R32F, w, h, GL_RED, GL_FLOAT, nullptr, 1);
    s.depth.parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    s.depth.parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    s.depth.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    s.depth.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    s.depth_rbo.storage(GL_DEPTH_COMPONENT24, w, h);

    s.fbo.bind();
    s.fbo.attach_texture(GL_COLOR_ATTACHMENT0, s.albedo);
    s.fbo.attach_texture(GL_COLOR_ATTACHMENT1, s.normal);
    s.fbo.attach_texture(GL_COLOR_ATTACHMENT2, s.position);
    s.fbo.attach_texture(GL_COLOR_ATTACHMENT3, s.surface);
    s.fbo.attach_texture(GL_COLOR_ATTACHMENT4, s.emissive);
    s.fbo.attach_texture(GL_COLOR_ATTACHMENT5, s.depth);
    s.fbo.attach_renderbuffer(GL_DEPTH_ATTACHMENT, s.depth_rbo);
    GLenum bufs[6] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2,
                      GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4, GL_COLOR_ATTACHMENT5};
    glDrawBuffers(6, bufs);
    if (!s.fbo.check()) {
        gllib::log(gllib::LogLevel::error, "SSRC G-buffer framebuffer incomplete");
    }
    gl::Framebuffer::unbind(gl::FramebufferType::both);
}

struct LevelParams {
    int spacing = 1;
    int rays = 1;
    int grid_w = 1;
    int grid_h = 1;
    int tex_w = 1;
    int tex_h = 1;
};

// Orbit camera: right-drag to orbit, scroll to zoom, WASD to pan the target.
void camera_control(gfx::Window& w, gfx::Camera& cam, float dt, bool allow, bool& captured) {
    static double px0 = 0, py0 = 0;
    static bool first = true;

    bool rmb = w.mouse_down(gfx::MouseButton::right);
    if (allow && rmb) {
        if (!captured) {
            glfwSetInputMode((GLFWwindow*)w.native_handle(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            captured = true;
            first = true;
        }
    } else if (captured) {
        glfwSetInputMode((GLFWwindow*)w.native_handle(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        captured = false;
    }

    if (captured) {
        double cx, cy;
        w.cursor_position(cx, cy);
        if (first) { px0 = cx; py0 = cy; first = false; }
        cam.orbit(float(cx - px0) * 0.008f, float(cy - py0) * 0.008f);
        px0 = cx;
        py0 = cy;
    }

    if (allow) {
        double scroll = w.scroll_delta();
        if (scroll != 0.0) cam.zoom(float(-scroll) * 0.05f);
    }

    float speed = 2.0f * dt;
    glm::vec3 fwd = glm::normalize(cam.target() - cam.position());
    glm::vec3 right = glm::normalize(glm::cross(fwd, glm::vec3(0, 1, 0)));
    glm::vec3 vel(0.0f);
    if (w.key_down(gfx::Key::w)) vel += fwd;
    if (w.key_down(gfx::Key::s)) vel -= fwd;
    if (w.key_down(gfx::Key::a)) vel += right;
    if (w.key_down(gfx::Key::d)) vel -= right;
    if (w.key_down(gfx::Key::space)) vel.y += 1;
    if (w.key_down(gfx::Key::shift)) vel.y -= 1;
    if (glm::length(vel) > 0.0f) vel = glm::normalize(vel) * speed;
    if (glm::length(vel) > 0.0f || captured) cam.set_target(cam.target() + vel);

    cam.orbit(0.0f, 0.0f);  // keep position synced to the target
}

} // namespace

// ===========================================================================
// Main
// ===========================================================================

int main() {
    gllib::log_to_stderr(gllib::LogLevel::info);
    gfx::Window window({"35 SSRC - Radiance Cascades", 1600, 900});

    // --- ImGui ---
    gfx::ImGuiOverlay gui;
    if (!gui.init(window)) {
        gllib::log(gllib::LogLevel::error, "ImGui init failed");
        return EXIT_FAILURE;
    }

    // --- Shaders ---
    auto make_program = [](const char* vs_src, const char* fs_src) -> gl::Program {
        gl::Shader vs(gl::ShaderType::vertex, vs_src);
        gl::Shader fs(gl::ShaderType::fragment, fs_src);
        gl::Program prog;
        prog.attach(vs);
        prog.attach(fs);
        if (!prog.link()) {
            gllib::log(gllib::LogLevel::error, prog.info_log().c_str());
        }
        return prog;
    };
    auto make_compute = [](const char* cs_src) -> gl::Program {
        gl::Shader cs(gl::ShaderType::compute, cs_src);
        gl::Program prog;
        prog.attach(cs);
        if (!prog.link()) {
            gllib::log(gllib::LogLevel::error, prog.info_log().c_str());
        }
        return prog;
    };

    gl::Program gbuf_prog = make_program(gbuf_vs, gbuf_fs);
    gl::Program build_prog = make_compute(build_cs);
    gl::Program gather_prog = make_compute(gather_cs);
    gl::Program display_prog = make_program(display_vs, display_fs);
    if (!gbuf_prog.linked() || !build_prog.linked() ||
        !gather_prog.linked() || !display_prog.linked()) {
        gllib::log(gllib::LogLevel::error, "Shader program compilation failed");
        return EXIT_FAILURE;
    }

    // --- Model ---
    gfx::Model model;
    if (!model.load("CornellBoxOriginal.glb")) {
        gllib::log(gllib::LogLevel::error, "Failed to load CornellBoxOriginal.glb");
        return EXIT_FAILURE;
    }
    gllib::logf(gllib::LogLevel::info,
                "Loaded: %zu meshes, %zu materials", model.mesh_count(), model.material_count());
    for (size_t i = 0; i < model.material_count(); ++i) {
        const auto& m = model.material_info(i);
        gllib::logf(gllib::LogLevel::info, "mat %zu: albedo(%.3f %.3f %.3f) emissive(%.3f %.3f %.3f)",
                    i, m.base_color_factor[0], m.base_color_factor[1], m.base_color_factor[2],
                    m.emissive_factor[0], m.emissive_factor[1], m.emissive_factor[2]);
    }

    // Rotate the box so its open face (+Y in model space) points at the camera.
    glm::mat4 model_mat = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0));

    glm::vec3 lo(FLT_MAX), hi(-FLT_MAX);
    for (int i = 0; i < int(model.mesh_count()); ++i) {
        glm::vec4 bs = model.mesh_bounding_sphere(i);
        glm::vec4 c = model_mat * glm::vec4(bs.x, bs.y, bs.z, 1.0f);
        for (int j = 0; j < 3; ++j) {
            lo[j] = std::min(lo[j], c[j] - bs.w);
            hi[j] = std::max(hi[j], c[j] + bs.w);
        }
    }
    glm::vec3 cam_target = (lo + hi) * 0.5f;
    float cam_dist = 0.5f * glm::length(hi - lo) + 3.0f;

    gfx::Camera cam;
    cam.perspective(45.0f, float(window.framebuffer_width()) / float(window.framebuffer_height()),
                    0.1f, 1000.0f);
    cam.look_at(cam_target + glm::vec3(0, 0, cam_dist), cam_target);

    // --- Fullscreen triangle (gl_VertexID) ---
    gl::VertexArray fsq_vao;

    // --- Tuning state ---
    int view_mode = getenv("SSRC_VIEW") ? std::atoi(getenv("SSRC_VIEW")) : 0;
    int debug_level = getenv("SSRC_DEBUG_LEVEL") ? std::atoi(getenv("SSRC_DEBUG_LEVEL")) : 1;
    int num_cascades = 6;
    int probe_spacing = 8;
    int rays_cap = 12;
    int ray_steps = 32;
    float base_interval = 0.02f;
    float res_scale = 0.5f;
    float gi_strength = 1.0f;
    float near_strength = 0.4f;
    float emissive_scale = 4.0f;
    float ray_bias = 0.005f;
    float thickness = 0.005f;
    glm::vec3 ambient_color(0.15f, 0.12f, 0.10f);
    float ambient_intensity = 0.5f;
    float exposure = 1.0f;
    int tonemap = 1;
    float gamma = 2.2f;
    const float far_plane = 1000.0f;

    // --- SSRC resources ---
    Ssrc ssrc;
    std::vector<gl::Texture> cascades;
    std::vector<LevelParams> lvl;
    int last_sw = -1, last_sh = -1, last_spacing = -1, last_cap = -1, last_n = -1;

    auto rebuild_cascades = [&]() {
        int sw = ssrc.w, sh = ssrc.h;
        lvl.resize(num_cascades);
        for (int c = 0; c < num_cascades; ++c) {
            int spacing = probe_spacing << c;
            int rays = std::min(spacing, rays_cap);
            int gw = std::max(1, (sw + spacing - 1) / spacing);
            int gh = std::max(1, (sh + spacing - 1) / spacing);
            lvl[c] = LevelParams{spacing, rays, gw, gh, gw * rays * rays, gh};
        }
        int n = num_cascades - 1;
        cascades.resize(n);
        for (int j = 0; j < n; ++j) {
            const LevelParams& lp = lvl[j + 1];
            gl::Texture& tex = cascades[j];
            tex.image_2d(0, GL_RGBA16F, lp.tex_w, lp.tex_h, GL_RGBA, GL_FLOAT, nullptr, 1);
            tex.parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            tex.parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            tex.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            tex.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
    };

    bool captured = false;
    int frame_count = 0;
    double last = window.time();

    while (!window.should_close()) {
        double now = window.time();
        float dt = float(now - last);
        last = now;

        window.poll_events();
        camera_control(window, cam, dt, !gui.wants_mouse(), captured);
        cam.set_aspect(float(window.framebuffer_width()) / float(window.framebuffer_height()));

        // --- (Re)create resources when size or cascade params change ---
        int fw = window.framebuffer_width(), fh = window.framebuffer_height();
        int sw = std::max(1, int(fw * res_scale));
        int sh = std::max(1, int(fh * res_scale));
        if (sw != last_sw || sh != last_sh) {
            create_ssrc_textures(ssrc, sw, sh);
            rebuild_cascades();
            last_sw = sw;
            last_sh = sh;
            last_spacing = probe_spacing;
            last_cap = rays_cap;
            last_n = num_cascades;
        } else if (probe_spacing != last_spacing || rays_cap != last_cap || num_cascades != last_n) {
            rebuild_cascades();
            last_spacing = probe_spacing;
            last_cap = rays_cap;
            last_n = num_cascades;
        }

        glm::mat4 vp = cam.view_projection();
        float seed = float(std::fmod(now, 1.0));

        // ===================================================================
        // 1. Geometry pass — 6 MRT G-buffer at ssrc resolution
        // ===================================================================
        ssrc.fbo.bind();
        gl::viewport(0, 0, sw, sh);
        gl::clear_color(0.0f, 0.0f, 0.0f, 1.0f);
        gl::clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        const float farv[4] = {far_plane, 0.0f, 0.0f, 0.0f};
        glClearBufferfv(GL_COLOR, 6, farv);

        gl::enable(GL_DEPTH_TEST);
        gl::depth_func(GL_LESS);

        gbuf_prog.use();
        auto g_loc = [&](const char* n) { return gbuf_prog.uniform_location(n); };
        GLint loc;
        loc = g_loc("u_view_proj"); if (loc >= 0) gbuf_prog.uniform_matrix4fv(loc, glm::value_ptr(vp));
        loc = g_loc("u_view");      if (loc >= 0) gbuf_prog.uniform_matrix4fv(loc, glm::value_ptr(cam.view()));
        loc = g_loc("u_model");     if (loc >= 0) gbuf_prog.uniform_matrix4fv(loc, glm::value_ptr(model_mat));
        loc = g_loc("u_emissive_scale");   if (loc >= 0) gbuf_prog.uniform1f(loc, emissive_scale);

        for (size_t i = 0; i < model.mesh_count(); ++i) {
            int mi = model.mesh_material(int(i));
            const auto& mat = model.material_info(size_t(mi >= 0 ? mi : 0));
            loc = g_loc("u_albedo");   if (loc >= 0) gbuf_prog.uniform3fv(loc, mat.base_color_factor);
            loc = g_loc("u_emissive"); if (loc >= 0) gbuf_prog.uniform3fv(loc, mat.emissive_factor);
            model.mesh(i).draw();
        }

        gl::disable(GL_DEPTH_TEST);
        gl::memory_barrier(GL_ALL_BARRIER_BITS);

        // ===================================================================
        // 2. Cascade build — coarsest to finest, one dispatch per cascade
        // ===================================================================
        build_prog.use();
        auto b_loc = [&](const char* n) { return build_prog.uniform_location(n); };
        loc = b_loc("u_view_proj"); if (loc >= 0) build_prog.uniform_matrix4fv(loc, glm::value_ptr(vp));
        loc = b_loc("u_view");      if (loc >= 0) build_prog.uniform_matrix4fv(loc, glm::value_ptr(cam.view()));
        loc = b_loc("u_ssrc_w");    if (loc >= 0) build_prog.uniform1i(loc, sw);
        loc = b_loc("u_ssrc_h");    if (loc >= 0) build_prog.uniform1i(loc, sh);
        loc = b_loc("u_far");       if (loc >= 0) build_prog.uniform1f(loc, far_plane);
        loc = b_loc("u_ray_bias");  if (loc >= 0) build_prog.uniform1f(loc, ray_bias);
        loc = b_loc("u_thickness"); if (loc >= 0) build_prog.uniform1f(loc, thickness);
        loc = b_loc("u_steps");     if (loc >= 0) build_prog.uniform1i(loc, ray_steps);
        loc = b_loc("u_seed");      if (loc >= 0) build_prog.uniform1f(loc, seed);
        loc = b_loc("u_ambient_color");      if (loc >= 0) build_prog.uniform3fv(loc, glm::value_ptr(ambient_color));
        loc = b_loc("u_ambient_intensity");  if (loc >= 0) build_prog.uniform1f(loc, ambient_intensity);

        ssrc.position.bind(0);
        ssrc.normal.bind(1);
        ssrc.surface.bind(2);
        ssrc.depth.bind_image(3, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);

        for (int c = num_cascades - 1; c >= 1; --c) {
            const LevelParams& cur = lvl[c];
            float istart = base_interval * std::pow(4.0f, float(c - 1));
            float iend = base_interval * std::pow(4.0f, float(c));
            int has_next = (c < num_cascades - 1) ? 1 : 0;

            cascades[c - 1].bind_image(4, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
            if (has_next) cascades[c].bind(5);

            loc = b_loc("u_grid_w");        if (loc >= 0) build_prog.uniform1i(loc, cur.grid_w);
            loc = b_loc("u_rays");          if (loc >= 0) build_prog.uniform1i(loc, cur.rays);
            loc = b_loc("u_spacing");       if (loc >= 0) build_prog.uniform1f(loc, float(cur.spacing));
            loc = b_loc("u_interval_start"); if (loc >= 0) build_prog.uniform1f(loc, istart);
            loc = b_loc("u_interval_end");  if (loc >= 0) build_prog.uniform1f(loc, iend);
            loc = b_loc("u_has_next");      if (loc >= 0) build_prog.uniform1i(loc, has_next);

            if (has_next) {
                const LevelParams& next = lvl[c + 1];
                loc = b_loc("u_next_rays");    if (loc >= 0) build_prog.uniform1i(loc, next.rays);
                loc = b_loc("u_next_spacing"); if (loc >= 0) build_prog.uniform1f(loc, float(next.spacing));
                loc = b_loc("u_next_grid_w");  if (loc >= 0) build_prog.uniform1i(loc, next.grid_w);
                loc = b_loc("u_next_tex_w");   if (loc >= 0) build_prog.uniform1i(loc, next.tex_w);
                loc = b_loc("u_next_tex_h");   if (loc >= 0) build_prog.uniform1i(loc, next.tex_h);
            }

            int groups_x = (cur.tex_w + 7) / 8;
            int groups_y = (cur.tex_h + 7) / 8;
            gl::dispatch_compute(groups_x, groups_y, 1);
            gl::memory_barrier(GL_ALL_BARRIER_BITS);
        }

        // ===================================================================
        // 3. Gather pass — integrate all cascades into a HDR target
        // ===================================================================
        gather_prog.use();
        auto g2_loc = [&](const char* n) { return gather_prog.uniform_location(n); };

        ssrc.albedo.bind_image(0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);
        ssrc.normal.bind_image(1, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);
        ssrc.emissive.bind_image(2, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);
        ssrc.position.bind_image(3, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);
        ssrc.depth.bind_image(4, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
        ssrc.color.bind_image(5, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

        ssrc.surface.bind(10);
        for (int j = 0; j < int(cascades.size()); ++j) cascades[j].bind(11 + j);

        loc = g2_loc("u_ssrc_w");        if (loc >= 0) gather_prog.uniform1i(loc, sw);
        loc = g2_loc("u_ssrc_h");        if (loc >= 0) gather_prog.uniform1i(loc, sh);
        loc = g2_loc("u_far");           if (loc >= 0) gather_prog.uniform1f(loc, far_plane);
        loc = g2_loc("u_num_cascades");  if (loc >= 0) gather_prog.uniform1i(loc, num_cascades);
        loc = g2_loc("u_gi_strength");   if (loc >= 0) gather_prog.uniform1f(loc, gi_strength);
        loc = g2_loc("u_near_strength"); if (loc >= 0) gather_prog.uniform1f(loc, near_strength);
        loc = g2_loc("u_view_mode");     if (loc >= 0) gather_prog.uniform1i(loc, view_mode);
        loc = g2_loc("u_debug_level");   if (loc >= 0) gather_prog.uniform1i(loc, debug_level);

        int base_rays = g2_loc("u_rays");
        int base_spacing = g2_loc("u_spacing");
        int base_grid = g2_loc("u_grid_w");
        int base_tex_w = g2_loc("u_tex_w");
        int base_tex_h = g2_loc("u_tex_h");
        for (int c = 0; c < num_cascades; ++c) {
            if (base_rays >= 0) gather_prog.uniform1i(base_rays + c, lvl[c].rays);
            if (base_spacing >= 0) gather_prog.uniform1i(base_spacing + c, lvl[c].spacing);
            if (base_grid >= 0) gather_prog.uniform1i(base_grid + c, lvl[c].grid_w);
            if (base_tex_w >= 0) gather_prog.uniform1i(base_tex_w + c, lvl[c].tex_w);
            if (base_tex_h >= 0) gather_prog.uniform1i(base_tex_h + c, lvl[c].tex_h);
        }

        int base_casc = g2_loc("u_cascades");
        if (base_casc >= 0) {
            int units[MAX_CASCADES];
            for (int j = 0; j < MAX_CASCADES; ++j) units[j] = 11 + j;
            for (int j = 0; j < MAX_CASCADES; ++j) gather_prog.uniform1i(base_casc + j, units[j]);
        }

        gl::dispatch_compute((sw + 7) / 8, (sh + 7) / 8, 1);
        gl::memory_barrier(GL_ALL_BARRIER_BITS);

        // ===================================================================
        // 4. Display pass — fullscreen triangle to the default framebuffer
        // ===================================================================
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        gl::viewport(0, 0, fw, fh);
        gl::clear_color(0.02f, 0.02f, 0.03f, 1.0f);
        gl::clear(GL_COLOR_BUFFER_BIT);

        display_prog.use();
        auto d_loc = [&](const char* n) { return display_prog.uniform_location(n); };
        ssrc.color.bind(0);
        loc = d_loc("u_color");    if (loc >= 0) display_prog.uniform1i(loc, 0);
        loc = d_loc("u_exposure"); if (loc >= 0) display_prog.uniform1f(loc, exposure);
        loc = d_loc("u_tonemap");  if (loc >= 0) display_prog.uniform1i(loc, tonemap);
        loc = d_loc("u_gamma");    if (loc >= 0) display_prog.uniform1f(loc, gamma);

        fsq_vao.bind();
        gl::draw_arrays(GL_TRIANGLES, 0, 3);

        // Screenshot: press P to save the current framebuffer.
        static bool prev_p = false;
        bool now_p = window.key_down(gfx::Key::p);
        if (now_p && !prev_p) gfx::screenshot("ssrc_screenshot.png");
        prev_p = now_p;
        if (const char* shot = getenv("SSRC_SHOT")) {
            static bool shot_done = false;
            if (!shot_done && frame_count == 45) {
                gfx::screenshot(shot);
                shot_done = true;
            }
        }

        // ===================================================================
        // 5. ImGui debug panel
        // ===================================================================
        gui.begin_frame();
        {
            ImGui::Begin("SSRC");
            ImGui::Text("FPS: %.1f  Frame: %.2f ms", 1.0f / std::max(dt, 1e-6f), dt * 1000.0f);
            ImGui::Text("SSRC res: %dx%d", sw, sh);
            ImGui::Separator();

            ImGui::Combo("View", &view_mode,
                         "Composite\0Albedo\0Normal\0Position\0Surface\0Emissive\0Depth\0Cascade\0");
            ImGui::SliderInt("Debug cascade", &debug_level, 0, num_cascades - 1);

            ImGui::Separator();
            ImGui::SliderInt("Cascades", &num_cascades, 2, MAX_CASCADES);
            ImGui::SliderInt("Probe spacing", &probe_spacing, 2, 64);
            ImGui::SliderInt("Rays cap", &rays_cap, 2, 16);
            ImGui::SliderFloat("Base interval", &base_interval, 0.001f, 1.0f, "%.3f");
            ImGui::SliderFloat("Res scale", &res_scale, 0.25f, 1.0f, "%.2f");
            ImGui::SliderInt("Ray steps", &ray_steps, 8, 64);

            ImGui::Separator();
            ImGui::SliderFloat("GI strength", &gi_strength, 0.0f, 4.0f);
            ImGui::SliderFloat("Near strength", &near_strength, 0.0f, 2.0f);
            ImGui::SliderFloat("Ray bias", &ray_bias, 0.0f, 0.1f, "%.4f");
            ImGui::SliderFloat("Thickness", &thickness, 0.0f, 0.1f, "%.4f");
            ImGui::SliderFloat("Emissive scale", &emissive_scale, 0.0f, 20.0f);

            ImGui::Separator();
            ImGui::ColorEdit3("Ambient", &ambient_color.x);
            ImGui::SliderFloat("Ambient intensity", &ambient_intensity, 0.0f, 2.0f);

            ImGui::Separator();
            ImGui::SliderFloat("Exposure", &exposure, 0.05f, 5.0f);
            ImGui::Combo("Tonemap", &tonemap, "Reinhard\0ACES\0");
            ImGui::SliderFloat("Gamma", &gamma, 1.0f, 3.0f);

            ImGui::Separator();
            float aw = ImGui::GetContentRegionAvail().x;
            if (aw < 64.0f) aw = 256.0f;
            ImGui::Image((ImTextureID)(intptr_t)ssrc.surface.handle(),
                         ImVec2(aw, aw * 0.5f), ImVec2(0, 1), ImVec2(1, 0));
            if (!cascades.empty()) {
                ImGui::Text("Cascade %d (packed)", debug_level);
                ImGui::Image((ImTextureID)(intptr_t)cascades[size_t(std::clamp(debug_level - 1, 0, int(cascades.size()) - 1))].handle(),
                             ImVec2(aw, aw * 0.3f), ImVec2(0, 1), ImVec2(1, 0));
            }
            ImGui::End();
        }
        gui.render();

        window.swap_buffers();
        window.poll_events();
        ++frame_count;
    }

    return EXIT_SUCCESS;
}
