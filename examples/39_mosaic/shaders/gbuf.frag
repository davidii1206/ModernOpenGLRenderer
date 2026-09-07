#version 460 core

#include "gbuffer.glsl"

layout(location = 0) out vec4 o_albedo;
layout(location = 1) out vec4 o_normal;
layout(location = 2) out vec4 o_emissive;
layout(location = 3) out vec2 o_motion;

in vec3 v_nrm;
in vec2 v_uv;
in vec4 v_clip;
in vec4 v_prev_clip;

layout(binding = 0) uniform sampler2D u_base_color;
layout(binding = 1) uniform sampler2D u_mr;
layout(binding = 2) uniform sampler2D u_emissive;

uniform vec4  u_base_color_factor;
uniform vec3  u_emissive_factor;
uniform float u_metallic_factor;
uniform float u_roughness_factor;
uniform int   u_has_base_color;
uniform int   u_has_mr;
uniform int   u_has_emissive;
uniform float u_alpha_cutoff;
uniform int   u_alpha_mask;

void main() {
    vec4 base = u_base_color_factor;
    if (u_has_base_color != 0) base *= texture(u_base_color, v_uv);
    if (u_alpha_mask != 0 && base.a < u_alpha_cutoff) discard;

    float metallic  = u_metallic_factor;
    float roughness = u_roughness_factor;
    if (u_has_mr != 0) {
        // glTF packs occlusion/roughness/metallic into R/G/B.
        vec3 mr = texture(u_mr, v_uv).rgb;
        roughness *= mr.g;
        metallic  *= mr.b;
    }

    vec3 emissive = u_emissive_factor;
    if (u_has_emissive != 0) emissive *= texture(u_emissive, v_uv).rgb;

    // Two-sided shading: flip the normal toward the viewer so back faces of
    // single-sided authored geometry (very common in Sponza) do not write
    // inverted normals that later break the surfel gather's plane tests.
    vec3 n = normalize(v_nrm);
    if (!gl_FrontFacing) n = -n;

    vec2 ndc      = v_clip.xy / v_clip.w;
    vec2 prev_ndc = v_prev_clip.xy / v_prev_clip.w;

    o_albedo   = vec4(base.rgb, 1.0);
    o_normal   = vec4(n, clamp(roughness, 0.02, 1.0));
    o_emissive = vec4(emissive, clamp(metallic, 0.0, 1.0));
    o_motion   = (prev_ndc - ndc) * 0.5;
}
