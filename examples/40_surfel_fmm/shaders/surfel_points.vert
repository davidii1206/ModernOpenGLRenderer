#version 460 core

// Raw point-cloud view of the surfel set. Unlike the gather this applies no
// weighting and no smoothing, so it is the check that the gather is not hiding
// something.

#include "surfel.glsl"

layout(std430, binding = 4) readonly buffer IrradIn { vec4 irrad[]; };

uniform mat4  u_view_proj;
uniform uint  u_count;
uniform int   u_color_mode;
uniform float u_point_scale;
uniform float u_irradiance_gain;
uniform float u_proj_scale;

out vec3 v_color;

vec3 turbo(float t) {
    t = clamp(t, 0.0, 1.0);
    return clamp(vec3(1.0 - abs(2.0 * t - 1.5), 1.0 - abs(2.0 * t - 1.0),
                      1.0 - abs(2.0 * t - 0.5)), 0.0, 1.0);
}

void main() {
    uint i = uint(gl_VertexID);
    vec4 pr = s_pos_rad[i];
    vec3 n  = surfel_normal(i);
    vec3 E  = irrad[i].rgb;

    switch (u_color_mode) {
        case 0: v_color = surfel_albedo(i); break;
        case 1: v_color = n * 0.5 + 0.5; break;
        case 2: v_color = turbo(pr.w * 200.0); break;
        case 3: v_color = surfel_emission(i); break;
        case 4: v_color = E * u_irradiance_gain; break;
        case 5: v_color = surfel_outgoing(i, E) * u_irradiance_gain; break;
        default: v_color = turbo(float(i) / float(max(u_count, 1u))); break;
    }

    vec4 clip = u_view_proj * vec4(pr.xyz, 1.0);
    gl_Position = clip;
    // clip.w is the view-space depth for a standard perspective matrix.
    gl_PointSize = clamp(2.0 * pr.w * u_point_scale * u_proj_scale / max(clip.w, 1e-4),
                         1.0, 64.0);
}
