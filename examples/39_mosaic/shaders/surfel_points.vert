#version 460 core

#include "surfel.glsl"
#include "sh.glsl"

// Debug point-cloud view of the baked object-space surfel sets. One draw per
// instance: gl_VertexID indexes the instance's slice of the shared set buffer,
// and the set is transformed into world space here exactly as S0 will.
layout(std430, binding = 0) readonly buffer Surfels { uvec4 surfels[]; };
layout(std430, binding = 1) readonly buffer Parents { uint parents[]; };
// Persistent per-source-surfel irradiance cache (4 x vec4, see sh.glsl).
layout(std430, binding = 2) readonly buffer ShCache { vec4 sh_cache[]; };

uniform mat4  u_view_proj;
uniform mat4  u_model;
uniform mat3  u_normal_mat;
uniform vec3  u_aabb_min;
uniform vec3  u_aabb_extent;
uniform float u_radius_scale;
uniform uint  u_base;          // first surfel of this set's selected LOD
uniform uint  u_lod;
uniform float u_point_scale;
uniform vec2  u_viewport;
uniform int   u_color_mode;
uniform float u_irradiance_gain;

out vec3 v_color;
out float v_view_z;

vec3 lod_color(uint lod) {
    if (lod == 0u) return vec3(0.35, 0.85, 0.45);
    if (lod == 1u) return vec3(0.35, 0.60, 0.95);
    if (lod == 2u) return vec3(0.95, 0.75, 0.30);
    return vec3(0.95, 0.35, 0.40);
}

void main() {
    uint idx = u_base + uint(gl_VertexID);
    uvec4 s = surfels[idx];

    vec3 p_obj = surfel_position(s, u_aabb_min, u_aabb_extent);
    float r = surfel_radius(s, u_radius_scale);
    vec3 n_obj = surfel_normal(s);
    uint flags = surfel_flags(s);

    vec4 world = u_model * vec4(p_obj, 1.0);
    vec3 n_world = normalize(u_normal_mat * n_obj);

    switch (u_color_mode) {
        case 1:  v_color = n_world * 0.5 + 0.5; break;
        case 2:  v_color = lod_color(u_lod); break;
        case 3:  v_color = (flags & SURFEL_EMISSIVE) != 0u
                            ? vec3(1.0, 0.85, 0.3) : vec3(0.12); break;
        case 4:  v_color = vec3(r / max(u_radius_scale, 1e-6)); break;
        case 5:  v_color = (flags & SURFEL_TWO_SIDED) != 0u
                            ? vec3(0.2, 0.9, 0.9) : vec3(0.15); break;
        case 6: {   // cached irradiance (S6 output), the cache seen directly
            SH1 sh = sh_load(sh_cache[idx * SH_STRIDE + 0u], sh_cache[idx * SH_STRIDE + 1u],
                             sh_cache[idx * SH_STRIDE + 2u], sh_cache[idx * SH_STRIDE + 3u]);
            v_color = sh_eval_irradiance(sh, n_world) * u_irradiance_gain;
            break;
        }
        case 7: {   // final: albedo * irradiance
            SH1 sh = sh_load(sh_cache[idx * SH_STRIDE + 0u], sh_cache[idx * SH_STRIDE + 1u],
                             sh_cache[idx * SH_STRIDE + 2u], sh_cache[idx * SH_STRIDE + 3u]);
            v_color = surfel_albedo(s) * sh_eval_irradiance(sh, n_world) * u_irradiance_gain;
            break;
        }
        case 8:     // update age: how stale this surfel's estimate is
            v_color = vec3(clamp(sh_cache[idx * SH_STRIDE + 1u].w / 64.0, 0.0, 1.0));
            break;
        default: v_color = surfel_albedo(s); break;
    }

    gl_Position = u_view_proj * world;
    v_view_z = gl_Position.w;
    // Screen-space size of the surfel disk, so the point cloud reads as a
    // covering of the surface rather than as a constant-size dot spray.
    float w = max(gl_Position.w, 1e-4);
    float scale = length(vec3(u_model[0]));   // uniform-scale assumption
    gl_PointSize = clamp(r * scale * u_point_scale * u_viewport.y / w, 1.0, 64.0);
}
