#version 460 core

#include "gbuffer.glsl"

layout(location = 0) out vec4 o_albedo;
layout(location = 1) out vec4 o_normal;
layout(location = 2) out vec4 o_emissive;

in vec3 v_nrm;
in vec2 v_uv;

layout(binding = 0) uniform sampler2D u_base_color_tex;
uniform uint  u_has_base_tex;
uniform vec4  u_base_color_factor;
uniform vec3  u_emissive_factor;   // KHR_materials_emissive_strength already folded in
uniform float u_metallic_factor;
uniform float u_roughness_factor;

void main() {
    // Flip the normal toward the viewer for double-sided geometry so the ceiling
    // panel's back face does not write an inverted normal that the gather's
    // plane test would then reject.
    vec3 n = normalize(v_nrm);
    if (!gl_FrontFacing) n = -n;

    // The factor alone was the whole story while the only scene was Cornell,
    // which is untextured. It is not for anything else: Sponza's materials are
    // white factors with all of the colour in the texture, so without this the
    // model renders grey and the bounce carries no colour to bleed.
    vec3 base = u_base_color_factor.rgb;
    if (u_has_base_tex != 0u) base *= texture(u_base_color_tex, v_uv).rgb;

    o_albedo   = vec4(base, 1.0);
    o_normal   = vec4(n, clamp(u_roughness_factor, 0.02, 1.0));
    o_emissive = vec4(u_emissive_factor, clamp(u_metallic_factor, 0.0, 1.0));
}
