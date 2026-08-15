// Example 35 — SSRC: Screen Space Radiance Cascades (radiance cascades GI).
//
// A hierarchical screen-space radiance field is built from the G-buffer. Each
// cascade level c has:
//   probe spacing : spacing0 * 2^c   (px)   -> 4x fewer probes per level
//   angular grid  : N0 * 2^c per side         -> 4x more directions per level
//   ray interval  : [t0*4^(c-1), t0*4^c]      -> non-overlapping, ratio 4
// Ray count per probe and probe count per level cancel, so every cascade does
// roughly the same amount of work.
//
// A probe ray is traced by marching in screen space between the projected start
// and end of its interval, comparing marched view depth against the depth
// buffer (thickness heuristic). Directions form a spherical grid around each
// probe (full azimuth x polar bands off the scene-facing axis), so probes reach
// scene content above, below, and beside them. On a hit the ray reads the
// outgoing radiance of that surface: emission plus the previous frame's
// indirect GI (temporal feedback), which makes bounces accumulate across
// frames. On a miss it merges at the probe's own position with the next coarser
// cascade (quadrilinear in space x angle), representing everything beyond this
// interval. Cascades are built coarsest-to-finest so each merge reads an
// already-resolved coarser level; the coarsest falls back to 0 (no sky). The
// final gather reads the resolved cascade 0 and computes the cosine-weighted
// hemisphere integral, storing the indirect term in a history buffer that feeds
// the next frame's build.
//
// The Cornell box has only an emissive ceiling light, so no point light or
// forward pass is needed: the emission buffer is the entire light source, and
// RC resolves both its "direct" illumination and bounces via the same trace.

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
constexpr int MAX_CASCADES = 8;

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
layout(location = 3) out vec4 out_source;
layout(location = 4) out float out_depth;

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
    vec3 view_n = normalize(mat3(u_view) * N);
    vec3 view_p = (u_view * vec4(v_pos, 1.0)).xyz;

    out_albedo   = vec4(u_albedo, 1.0);
    out_normal   = vec4(octahedral(view_n), 0.0, 1.0);
    out_position = vec4(view_p, 1.0);
    out_source   = vec4(u_emissive * u_emissive_scale, 1.0);
    out_depth    = -view_p.z;
}
)";

const char* build_cs = R"(
#version 460 core
layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0) uniform sampler2D u_pos_tex;     // view-space position, a = surface
layout(binding = 1) uniform sampler2D u_depth_tex;   // R32F linear view depth
layout(binding = 2) uniform sampler2D u_source_tex;  // outgoing radiance (emission)
layout(binding = 3) uniform sampler2D u_gi_tex;      // previous frame's indirect GI (feedback)
layout(rgba16f, binding = 4) uniform writeonly image2D u_out;
layout(binding = 5) uniform sampler2D u_next_tex;    // coarser cascade atlas

uniform mat4 u_proj;
uniform vec2 u_res;
uniform float u_far;
uniform float u_feedback;

uniform int   u_N;             // angular grid side for this cascade
uniform float u_spacing;       // probe spacing (px at ssrc res)
uniform int   u_probes_x;
uniform int   u_probes_y;
uniform int   u_tex_w;
uniform int   u_tex_h;
uniform float u_t0;            // interval start
uniform float u_t1;            // interval end
uniform int   u_has_next;
uniform int   u_next_N;
uniform float u_next_spacing;
uniform int   u_next_probes_x;
uniform int   u_next_probes_y;
uniform float u_thickness;
uniform float u_skip_px;
uniform float u_step_px;

// Spherical direction grid nested per cascade, spanning the FULL sphere:
// phi in [0,2pi) with N cells, theta in [0,pi] with N bands. Doubling N halves
// both angular steps, so the 2x2 coarser angular block nests exactly under
// cell (ai, aj). Covering the full sphere lets surfaces facing the camera
// receive light (their incoming hemisphere includes +z directions).
vec3 dir_from_angle(int ai, int aj, int N) {
    float phi = (float(ai) + 0.5) / float(N) * 6.2831853;
    float theta = (float(aj) + 0.5) / float(N) * 3.1415927;
    float s = sin(theta);
    return normalize(vec3(s * cos(phi), s * sin(phi), cos(theta)));
}

vec2 uv_from_view(vec3 p) {
    vec4 clip = u_proj * vec4(p, 1.0);
    return clip.xy / clip.w * 0.5 + 0.5;
}

// Quadrilinear (bilinear space x bilinear angle) sample of the coarser cascade
// at the current probe position. Angular cells map to the coarser grid such
// that for doubled N cell (ai, aj) lands on the center of the 2x2 block.
vec3 sample_next(vec2 pos, int ai, int aj) {
    vec2 fp = pos / u_next_spacing - 0.5;
    vec2 p0 = floor(fp);
    vec2 f = fp - p0;
    vec2 fa = (vec2(float(ai), float(aj)) + 0.5) * (float(u_next_N) / float(u_N)) - 0.5;
    vec2 a0 = floor(fa);
    vec2 af = fa - a0;
    vec3 acc = vec3(0.0);
    for (int dx = 0; dx < 2; ++dx)
    for (int dy = 0; dy < 2; ++dy)
    for (int ax = 0; ax < 2; ++ax)
    for (int ay = 0; ay < 2; ++ay) {
        ivec2 pi = clamp(ivec2(p0) + ivec2(dx, dy),
                         ivec2(0), ivec2(u_next_probes_x - 1, u_next_probes_y - 1));
        ivec2 ai = clamp(ivec2(a0) + ivec2(ax, ay),
                         ivec2(0), ivec2(u_next_N - 1));
        vec3 v = texelFetch(u_next_tex,
                            ivec2(pi.x * u_next_N + ai.x, pi.y * u_next_N + ai.y), 0).rgb;
        float w = ((dx == 0) ? (1.0 - f.x) : f.x)
                * ((dy == 0) ? (1.0 - f.y) : f.y)
                * ((ax == 0) ? (1.0 - af.x) : af.x)
                * ((ay == 0) ? (1.0 - af.y) : af.y);
        acc += v * w;
    }
    return acc;
}

// March the ray from u_t0 to u_t1 (view-space distances) in screen space.
bool march(vec3 origin, vec3 dir, out vec3 rad, out vec2 hit_uv) {
    vec3 s = origin + dir * u_t0;
    vec3 e = origin + dir * u_t1;
    vec2 s_uv = uv_from_view(s);
    vec2 e_uv = uv_from_view(e);
    vec2 d_uv = e_uv - s_uv;
    float len = length(d_uv * u_res);
    if (len < 1e-5) return false;

    int steps = clamp(int(ceil(len / u_step_px)), 2, 256);
    float start = clamp(u_skip_px / max(len, 1e-6), 0.0, 0.85);
    float inv_d0 = 1.0 / max(-s.z, 1e-6);
    float inv_d1 = 1.0 / max(-e.z, 1e-6);
    float prev_ray_d = -s.z;

    for (int i = 0; i < steps; ++i) {
        float t = start + (float(i) + 0.5) / float(steps) * (1.0 - start);
        vec2 uv = s_uv + d_uv * t;
        if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0) return false;

        float ray_d = 1.0 / mix(inv_d0, inv_d1, t);
        float surf_d = texture(u_depth_tex, uv).r;
        if (surf_d > 0.001 && surf_d < u_far - 1.0) {
            bool crossed = (prev_ray_d <= surf_d && ray_d >= surf_d);
            bool near = abs(ray_d - surf_d) < u_thickness;
            if (crossed || near) {
                rad = texture(u_source_tex, uv).rgb
                    + texture(u_gi_tex, uv).rgb * u_feedback;
                hit_uv = uv;
                return true;
            }
        }
        prev_ray_d = ray_d;
    }
    return false;
}

void main() {
    ivec2 t = ivec2(gl_GlobalInvocationID.xy);
    if (t.x >= u_tex_w || t.y >= u_tex_h) return;

    int px = t.x / u_N;
    int py = t.y / u_N;
    int ai = t.x % u_N;
    int aj = t.y % u_N;

    vec2 probe_px = (vec2(px, py) + 0.5) * u_spacing;
    ivec2 pt = ivec2(clamp(round(probe_px), vec2(0.0), u_res - 1.0));
    vec4 P = texelFetch(u_pos_tex, pt, 0);
    if (P.a < 0.5) {
        imageStore(u_out, t, vec4(0.0, 0.0, 0.0, 0.0));
        return;
    }

    vec3 origin = P.xyz;
    vec3 dir = dir_from_angle(ai, aj, u_N);

    vec3 rad;
    vec2 hit_uv;
    if (march(origin, dir, rad, hit_uv)) {
        imageStore(u_out, t, vec4(rad, 1.0));
    } else if (u_has_next == 1) {
        vec3 m = sample_next(probe_px, ai, aj);
        imageStore(u_out, t, vec4(m, 1.0));
    } else {
        imageStore(u_out, t, vec4(0.0, 0.0, 0.0, 0.0));
    }
}
)";

const char* gather_cs = R"(
#version 460 core
layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba16f, binding = 0) uniform readonly image2D u_albedo_tex;
layout(rgba16f, binding = 1) uniform readonly image2D u_normal_tex;
layout(rgba16f, binding = 2) uniform readonly image2D u_source_tex;
layout(rgba16f, binding = 3) uniform readonly image2D u_position_tex;
layout(r32f, binding = 4) uniform readonly image2D u_depth_tex;
layout(rgba16f, binding = 5) uniform writeonly image2D u_out;
layout(binding = 6) uniform sampler2D u_cascade0;
layout(binding = 7) uniform sampler2D u_cascade_dbg;
layout(rgba16f, binding = 7) uniform writeonly image2D u_gi_out;   // indirect-only history

uniform vec2 u_res;
uniform float u_far;
uniform int   u_N;             // cascade 0 angular grid side
uniform float u_spacing;       // cascade 0 probe spacing
uniform int   u_probes_x;
uniform int   u_probes_y;
uniform float u_gi_strength;
uniform float u_gi_clamp;
uniform int   u_view_mode;
uniform int   u_debug_level;
uniform int   u_dbg_N;
uniform float u_dbg_spacing;
uniform int   u_dbg_probes_x;
uniform int   u_dbg_probes_y;

vec3 oct_decode(vec2 e) {
    e = e * 2.0 - 1.0;
    vec3 n = vec3(e, 1.0);
    n.z = 1.0 - abs(n.x) - abs(n.y);
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(n.yx)) * mix(vec2(-1.0), vec2(1.0), step(0.0, n.xy));
    }
    return normalize(n);
}

vec3 dir_from_angle(int ai, int aj, int N) {
    float phi = (float(ai) + 0.5) / float(N) * 6.2831853;
    float theta = (float(aj) + 0.5) / float(N) * 3.1415927;
    float s = sin(theta);
    return normalize(vec3(s * cos(phi), s * sin(phi), cos(theta)));
}

// Bilinear probe sample of the resolved cascade 0 at a pixel position.
vec3 c0_sample(vec2 pos, int ai, int aj) {
    vec2 fp = pos / u_spacing - 0.5;
    vec2 p0 = floor(fp);
    vec2 f = fp - p0;
    vec3 acc = vec3(0.0);
    for (int dx = 0; dx < 2; ++dx)
    for (int dy = 0; dy < 2; ++dy) {
        ivec2 pi = clamp(ivec2(p0) + ivec2(dx, dy),
                         ivec2(0), ivec2(u_probes_x - 1, u_probes_y - 1));
        vec3 v = texelFetch(u_cascade0, ivec2(pi.x * u_N + ai, pi.y * u_N + aj), 0).rgb;
        float w = ((dx == 0) ? (1.0 - f.x) : f.x) * ((dy == 0) ? (1.0 - f.y) : f.y);
        acc += v * w;
    }
    return acc;
}

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    if (px.x >= int(u_res.x) || px.y >= int(u_res.y)) return;

    vec4 alb4 = imageLoad(u_albedo_tex, px);
    if (alb4.a < 0.5) {
        imageStore(u_out, px, vec4(0.0, 0.0, 0.0, 1.0));
        imageStore(u_gi_out, px, vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }
    vec3 albedo = alb4.rgb;
    vec3 n = oct_decode(imageLoad(u_normal_tex, px).xy);

    if (u_view_mode == 1) { imageStore(u_out, px, vec4(albedo, 1.0)); imageStore(u_gi_out, px, vec4(0.0, 0.0, 0.0, 1.0)); return; }
    if (u_view_mode == 2) { imageStore(u_out, px, vec4(n * 0.5 + 0.5, 1.0)); imageStore(u_gi_out, px, vec4(0.0, 0.0, 0.0, 1.0)); return; }
    if (u_view_mode == 3) {
        vec3 p = clamp(imageLoad(u_position_tex, px).rgb * 0.5 + 0.5, 0.0, 1.0);
        imageStore(u_out, px, vec4(p, 1.0));
        imageStore(u_gi_out, px, vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }
    if (u_view_mode == 4) {
        imageStore(u_out, px, vec4(imageLoad(u_source_tex, px).rgb, 1.0));
        imageStore(u_gi_out, px, vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }
    if (u_view_mode == 5) {
        float d = imageLoad(u_depth_tex, px).r / u_far;
        imageStore(u_out, px, vec4(vec3(d), 1.0));
        imageStore(u_gi_out, px, vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }
    if (u_view_mode == 6) {
        vec2 puv = (vec2(px) + 0.5) / u_res;
        vec2 fp = puv * u_res / u_dbg_spacing - 0.5;
        ivec2 probe = clamp(ivec2(floor(fp)),
                            ivec2(0), ivec2(u_dbg_probes_x - 1, u_dbg_probes_y - 1));
        int ai = u_dbg_N / 2;
        int aj = u_dbg_N / 2;
        vec3 c = texelFetch(u_cascade_dbg,
                            ivec2(probe.x * u_dbg_N + ai, probe.y * u_dbg_N + aj), 0).rgb;
        imageStore(u_out, px, vec4(c, 1.0));
        imageStore(u_gi_out, px, vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }

    vec3 E = vec3(0.0);
    float wsum = 0.0;
    vec2 pos = (vec2(px) + 0.5) / u_res * u_res;
    for (int aj = 0; aj < u_N; ++aj)
    for (int ai = 0; ai < u_N; ++ai) {
        vec3 dir = dir_from_angle(ai, aj, u_N);
        float c = dot(n, dir);
        if (c <= 0.0) continue;
        E += c0_sample(pos, ai, aj) * c;
        wsum += c;
    }
    vec3 gi = (wsum > 1e-4) ? E / wsum : vec3(0.0);
    vec3 indirect = albedo * gi * u_gi_strength;
    vec3 final = indirect + imageLoad(u_source_tex, px).rgb;
    imageStore(u_out, px, vec4(final, 1.0));
    imageStore(u_gi_out, px, vec4(min(indirect, vec3(u_gi_clamp)), 1.0));
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
    gl::Texture source{gl::TextureType::tex_2d};
    gl::Texture gi{gl::TextureType::tex_2d};
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
    tex2(s.source);
    tex2(s.gi);
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
    s.fbo.attach_texture(GL_COLOR_ATTACHMENT3, s.source);
    s.fbo.attach_texture(GL_COLOR_ATTACHMENT4, s.depth);
    s.fbo.attach_renderbuffer(GL_DEPTH_ATTACHMENT, s.depth_rbo);
    GLenum bufs[5] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2,
                      GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4};
    glDrawBuffers(5, bufs);
    if (!s.fbo.check()) {
        gllib::log(gllib::LogLevel::error, "SSRC G-buffer framebuffer incomplete");
    }
    gl::Framebuffer::unbind(gl::FramebufferType::both);

    const float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    glClearTexImage(s.gi.handle(), 0, GL_RGBA, GL_FLOAT, zero);
}

struct LevelParams {
    int spacing = 1;
    int N = 1;
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
    float cam_dist = 0.5f * glm::length(hi - lo) + 1.8f;

    gfx::Camera cam;
    cam.perspective(45.0f, float(window.framebuffer_width()) / float(window.framebuffer_height()),
                    0.1f, 1000.0f);
    glm::vec3 cam_offset = glm::vec3(0, 0, cam_dist);
    if (getenv("SSRC_CAM_X") && getenv("SSRC_CAM_Y") && getenv("SSRC_CAM_Z")) {
        cam_offset = glm::vec3(std::atof(getenv("SSRC_CAM_X")),
                               std::atof(getenv("SSRC_CAM_Y")),
                               std::atof(getenv("SSRC_CAM_Z")));
    }
    cam.look_at(cam_target + cam_offset, cam_target);

    gllib::logf(gllib::LogLevel::info, "SSRC camera target(%.2f %.2f %.2f) dist %.2f",
                cam_target.x, cam_target.y, cam_target.z, cam_dist);

    // --- Fullscreen triangle (gl_VertexID) ---
    gl::VertexArray fsq_vao;

    // --- Tuning state ---
    int view_mode = getenv("SSRC_VIEW") ? std::atoi(getenv("SSRC_VIEW")) : 0;
    int debug_level = getenv("SSRC_DEBUG_LEVEL") ? std::atoi(getenv("SSRC_DEBUG_LEVEL")) : 1;
    int num_cascades = 4;
    int probe_spacing = 8;
    int rays_base = 8;
    float base_interval = 0.1f;
    float res_scale = 0.5f;
    float gi_strength = 1.5f;
    float feedback = 0.9f;
    float gi_clamp = 200.0f;
    float emissive_scale = 30.0f;
    float thickness = 0.01f;
    float skip_px = 3.0f;
    float step_px = 2.0f;
    float exposure = 1.0f;
    int tonemap = 1;
    float gamma = 2.2f;
    const float far_plane = 1000.0f;

    // --- SSRC resources ---
    Ssrc ssrc;
    std::vector<gl::Texture> cascades;
    std::vector<LevelParams> lvl;
    int last_sw = -1, last_sh = -1, last_spacing = -1, last_rays = -1, last_n = -1;

    auto rebuild_cascades = [&]() {
        int sw = ssrc.w, sh = ssrc.h;
        lvl.resize(num_cascades);
        for (int c = 0; c < num_cascades; ++c) {
            int spacing = probe_spacing << c;
            int N = rays_base << c;
            int gw = std::max(1, (sw + spacing - 1) / spacing);
            int gh = std::max(1, (sh + spacing - 1) / spacing);
            lvl[c] = LevelParams{spacing, N, gw, gh, gw * N, gh * N};
        }
        cascades.resize(num_cascades);
        for (int c = 0; c < num_cascades; ++c) {
            const LevelParams& lp = lvl[c];
            gl::Texture& tex = cascades[c];
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
            last_rays = rays_base;
            last_n = num_cascades;
        } else if (probe_spacing != last_spacing || rays_base != last_rays ||
                   num_cascades != last_n) {
            rebuild_cascades();
            last_spacing = probe_spacing;
            last_rays = rays_base;
            last_n = num_cascades;
        }

        glm::mat4 vp = cam.view_projection();
        glm::mat4 proj = cam.projection();

        // ===================================================================
        // 1. Geometry pass — G-buffer + source (emission) at ssrc resolution
        // ===================================================================
        ssrc.fbo.bind();
        gl::viewport(0, 0, sw, sh);
        gl::clear_color(0.0f, 0.0f, 0.0f, 0.0f);
        gl::clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        const float farv[4] = {far_plane, 0.0f, 0.0f, 0.0f};
        glClearBufferfv(GL_COLOR, 4, farv);

        gl::enable(GL_DEPTH_TEST);
        gl::depth_func(GL_LESS);

        gbuf_prog.use();
        auto g_loc = [&](const char* n) { return gbuf_prog.uniform_location(n); };
        GLint loc;
        loc = g_loc("u_view_proj"); if (loc >= 0) gbuf_prog.uniform_matrix4fv(loc, glm::value_ptr(vp));
        loc = g_loc("u_view");      if (loc >= 0) gbuf_prog.uniform_matrix4fv(loc, glm::value_ptr(cam.view()));
        loc = g_loc("u_model");     if (loc >= 0) gbuf_prog.uniform_matrix4fv(loc, glm::value_ptr(model_mat));
        loc = g_loc("u_emissive_scale"); if (loc >= 0) gbuf_prog.uniform1f(loc, emissive_scale);

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
        // 2. Cascade build — coarsest to finest; each pass traces its interval
        //    and merges with the already-resolved next coarser cascade.
        // ===================================================================
        build_prog.use();
        auto b_loc = [&](const char* n) { return build_prog.uniform_location(n); };
        loc = b_loc("u_proj");      if (loc >= 0) build_prog.uniform_matrix4fv(loc, glm::value_ptr(proj));
        loc = b_loc("u_res");       if (loc >= 0) build_prog.uniform2f(loc, float(sw), float(sh));
        loc = b_loc("u_far");       if (loc >= 0) build_prog.uniform1f(loc, far_plane);
        loc = b_loc("u_thickness"); if (loc >= 0) build_prog.uniform1f(loc, thickness);
        loc = b_loc("u_skip_px");   if (loc >= 0) build_prog.uniform1f(loc, skip_px);
        loc = b_loc("u_step_px");   if (loc >= 0) build_prog.uniform1f(loc, step_px);
        loc = b_loc("u_feedback");  if (loc >= 0) build_prog.uniform1f(loc, feedback);

        ssrc.position.bind(0);
        ssrc.depth.bind(1);
        ssrc.source.bind(2);
        ssrc.gi.bind(3);

        for (int c = num_cascades - 1; c >= 0; --c) {
            const LevelParams& cur = lvl[c];
            float t0 = (c == 0) ? 0.0f : base_interval * std::pow(4.0f, float(c - 1));
            float t1 = base_interval * std::pow(4.0f, float(c));
            int has_next = (c < num_cascades - 1) ? 1 : 0;

            cascades[c].bind_image(4, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
            if (has_next) cascades[c + 1].bind(5);

            loc = b_loc("u_N");             if (loc >= 0) build_prog.uniform1i(loc, cur.N);
            loc = b_loc("u_spacing");       if (loc >= 0) build_prog.uniform1f(loc, float(cur.spacing));
            loc = b_loc("u_probes_x");      if (loc >= 0) build_prog.uniform1i(loc, cur.grid_w);
            loc = b_loc("u_probes_y");      if (loc >= 0) build_prog.uniform1i(loc, cur.grid_h);
            loc = b_loc("u_tex_w");         if (loc >= 0) build_prog.uniform1i(loc, cur.tex_w);
            loc = b_loc("u_tex_h");         if (loc >= 0) build_prog.uniform1i(loc, cur.tex_h);
            loc = b_loc("u_t0");            if (loc >= 0) build_prog.uniform1f(loc, t0);
            loc = b_loc("u_t1");            if (loc >= 0) build_prog.uniform1f(loc, t1);
            loc = b_loc("u_has_next");      if (loc >= 0) build_prog.uniform1i(loc, has_next);

            if (has_next) {
                const LevelParams& next = lvl[c + 1];
                loc = b_loc("u_next_N");        if (loc >= 0) build_prog.uniform1i(loc, next.N);
                loc = b_loc("u_next_spacing");  if (loc >= 0) build_prog.uniform1f(loc, float(next.spacing));
                loc = b_loc("u_next_probes_x"); if (loc >= 0) build_prog.uniform1i(loc, next.grid_w);
                loc = b_loc("u_next_probes_y"); if (loc >= 0) build_prog.uniform1i(loc, next.grid_h);
            }

            int groups_x = (cur.tex_w + 7) / 8;
            int groups_y = (cur.tex_h + 7) / 8;
            gl::dispatch_compute(groups_x, groups_y, 1);
            gl::memory_barrier(GL_ALL_BARRIER_BITS);
        }

        // ===================================================================
        // 3. Final gather — cosine-weighted hemisphere integral from cascade 0
        // ===================================================================
        gather_prog.use();
        auto g2_loc = [&](const char* n) { return gather_prog.uniform_location(n); };

        ssrc.albedo.bind_image(0, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);
        ssrc.normal.bind_image(1, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);
        ssrc.source.bind_image(2, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);
        ssrc.position.bind_image(3, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);
        ssrc.depth.bind_image(4, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
        ssrc.color.bind_image(5, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
        ssrc.gi.bind_image(7, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

        cascades[0].bind(6);
        cascades[size_t(std::clamp(debug_level, 0, num_cascades - 1))].bind(7);

        const LevelParams& c0 = lvl[0];
        const LevelParams& dbg = lvl[std::clamp(debug_level, 0, num_cascades - 1)];
        loc = g2_loc("u_res");         if (loc >= 0) gather_prog.uniform2f(loc, float(sw), float(sh));
        loc = g2_loc("u_far");         if (loc >= 0) gather_prog.uniform1f(loc, far_plane);
        loc = g2_loc("u_N");           if (loc >= 0) gather_prog.uniform1i(loc, c0.N);
        loc = g2_loc("u_spacing");     if (loc >= 0) gather_prog.uniform1f(loc, float(c0.spacing));
        loc = g2_loc("u_probes_x");    if (loc >= 0) gather_prog.uniform1i(loc, c0.grid_w);
        loc = g2_loc("u_probes_y");    if (loc >= 0) gather_prog.uniform1i(loc, c0.grid_h);
        loc = g2_loc("u_gi_strength"); if (loc >= 0) gather_prog.uniform1f(loc, gi_strength);
        loc = g2_loc("u_gi_clamp");    if (loc >= 0) gather_prog.uniform1f(loc, gi_clamp);
        loc = g2_loc("u_view_mode");   if (loc >= 0) gather_prog.uniform1i(loc, view_mode);
        loc = g2_loc("u_debug_level"); if (loc >= 0) gather_prog.uniform1i(loc, debug_level);
        loc = g2_loc("u_dbg_N");          if (loc >= 0) gather_prog.uniform1i(loc, dbg.N);
        loc = g2_loc("u_dbg_spacing");    if (loc >= 0) gather_prog.uniform1f(loc, float(dbg.spacing));
        loc = g2_loc("u_dbg_probes_x");   if (loc >= 0) gather_prog.uniform1i(loc, dbg.grid_w);
        loc = g2_loc("u_dbg_probes_y");   if (loc >= 0) gather_prog.uniform1i(loc, dbg.grid_h);

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
                         "Composite\0Albedo\0Normal\0Position\0Source\0Depth\0Cascade\0");
            ImGui::SliderInt("Debug cascade", &debug_level, 0, num_cascades - 1);

            ImGui::Separator();
            ImGui::SliderInt("Cascades", &num_cascades, 2, MAX_CASCADES);
            ImGui::SliderInt("Probe spacing", &probe_spacing, 2, 64);
            ImGui::SliderInt("Rays per side", &rays_base, 2, 16);
            ImGui::SliderFloat("Base interval", &base_interval, 0.001f, 1.0f, "%.3f");
            ImGui::SliderFloat("Res scale", &res_scale, 0.25f, 1.0f, "%.2f");
            ImGui::SliderFloat("Step px", &step_px, 0.5f, 8.0f, "%.2f");
            ImGui::SliderFloat("Skip px", &skip_px, 0.0f, 16.0f, "%.1f");

            ImGui::Separator();
            ImGui::SliderFloat("GI strength", &gi_strength, 0.0f, 4.0f);
            ImGui::SliderFloat("Feedback", &feedback, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("GI clamp", &gi_clamp, 0.0f, 500.0f);
            ImGui::SliderFloat("Thickness", &thickness, 0.0f, 0.2f, "%.4f");
            ImGui::SliderFloat("Emissive scale", &emissive_scale, 0.0f, 50.0f);

            ImGui::Separator();
            ImGui::SliderFloat("Exposure", &exposure, 0.05f, 5.0f);
            ImGui::Combo("Tonemap", &tonemap, "Reinhard\0ACES\0");
            ImGui::SliderFloat("Gamma", &gamma, 1.0f, 3.0f);

            ImGui::Separator();
            float aw = ImGui::GetContentRegionAvail().x;
            if (aw < 64.0f) aw = 256.0f;
            ImGui::Image((ImTextureID)(intptr_t)ssrc.source.handle(),
                         ImVec2(aw, aw * 0.5f), ImVec2(0, 1), ImVec2(1, 0));
            ImGui::Text("GI history (indirect, feedback input)");
            ImGui::Image((ImTextureID)(intptr_t)ssrc.gi.handle(),
                         ImVec2(aw, aw * 0.5f), ImVec2(0, 1), ImVec2(1, 0));
            if (!cascades.empty()) {
                ImGui::Text("Cascade %d (packed atlas)", debug_level);
                ImGui::Image((ImTextureID)(intptr_t)
                                 cascades[size_t(std::clamp(debug_level, 0, num_cascades - 1))].handle(),
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
