#version 460 core

// Tonemap, debug views, and the comparison against the path-traced references.
//
// Unchanged from example 40 apart from the irradiance source and one view mode,
// deliberately: the two examples estimate the same quantity by different means,
// and a shared display path means a split view between them compares the
// estimators and not two different tone curves. The AgX gap example 40 documents
// applies here word for word -- both references are Blender Cycles renders under
// AgX, none of the curves below is AgX, so the numeric diff is approximate and
// most visibly so on the emitter panel itself.

#include "gbuffer.glsl"
#include "brdf.glsl"
#include "tonemap.glsl"

in vec2 v_uv;
out vec4 frag_color;

layout(binding = 0) uniform sampler2D u_albedo;
layout(binding = 1) uniform sampler2D u_normal;
layout(binding = 2) uniform sampler2D u_emissive;
layout(binding = 3) uniform sampler2D u_depth;
layout(binding = 4) uniform sampler2D u_recon;      // (E, 1), upsampled from the GI grid
layout(binding = 5) uniform sampler2D u_gt_direct;
layout(binding = 6) uniform sampler2D u_gt_full;

uniform int   u_view_mode;
uniform float u_exposure;
uniform float u_irradiance_gain;
uniform float u_diff_gain;
uniform float u_split_x;
uniform int   u_gt_index;
uniform int   u_tonemap;
uniform mat4  u_inv_view_proj;
uniform vec3  u_scene_min;
uniform vec3  u_scene_extent;

// Linear light -> what the monitor shows. The references are already tonemapped
// by the path tracer, so this is the space the comparison has to happen in.
vec3 to_display(vec3 linear) {
    return sgi_tonemap(linear * u_exposure, u_tonemap);
}

vec3 heat(float t) {
    t = clamp(t, 0.0, 1.0);
    return clamp(vec3(1.5 - abs(4.0 * t - 3.0), 1.5 - abs(4.0 * t - 2.0),
                      1.5 - abs(4.0 * t - 1.0)), 0.0, 1.0);
}

void main() {
    vec2 uv = v_uv;

    vec4  alb   = texture(u_albedo, uv);
    vec4  nrm   = texture(u_normal, uv);
    vec4  emi   = texture(u_emissive, uv);
    float depth = texture(u_depth, uv).r;
    vec4  acc   = texture(u_recon, uv);

    // upsample.comp writes (E, 1), so this divide is a no-op on a reached pixel
    // and keeps an unreached one at zero rather than exploding on a ~0 alpha.
    vec3 E = acc.a > 1e-6 ? acc.rgb / acc.a : vec3(0.0);

    // L_out = L_e + albedo * E / PI, through the same helper the compute
    // kernels use -- see common/brdf.glsl.
    vec3 lit = sgi_outgoing(emi.rgb, alb.rgb, E);

    // stb_image loads top-down and gfx::Texture does not flip, so the reference
    // arrives with its first row at v = 0. Flip to match our own framebuffer.
    vec2 guv = vec2(uv.x, 1.0 - uv.y);
    vec3 gt = (u_gt_index == 0 ? texture(u_gt_direct, guv) : texture(u_gt_full, guv)).rgb;

    vec3 ours = to_display(lit);

    vec3 color;
    bool raw = true;   // already in display space, skip the tonemap below

    switch (u_view_mode) {
        case 0: color = alb.rgb; break;
        case 1: color = nrm.xyz * 0.5 + 0.5; break;
        case 2: color = emi.rgb; raw = false; break;
        case 3: color = vec3(pow(depth, 64.0)); break;
        case 4: color = (world_from_depth(depth, uv, u_inv_view_proj) - u_scene_min)
                        / max(u_scene_extent, vec3(1e-6)); break;
        case 5: color = E * u_irradiance_gain; raw = false; break;
        case 6: color = lit - emi.rgb; raw = false; break;      // bounce only
        case 7: color = ours; break;
        case 8: color = heat(dot(E, vec3(0.3333)) * u_irradiance_gain); break;
        case 9: color = gt; break;
        case 10:
            color = uv.x < u_split_x ? ours : gt;
            if (abs(uv.x - u_split_x) < 0.0015) color = vec3(1.0, 1.0, 0.0);
            break;
        default: {
            vec3 d3 = abs(ours - gt) * u_diff_gain;
            color = heat(dot(d3, vec3(0.3333)));
            break;
        }
    }

    if (!raw) color = to_display(color);
    frag_color = vec4(color, 1.0);
}
