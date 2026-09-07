#version 460 core

#include "gbuffer.glsl"

const int OCC_RES = 128;
layout(binding = 7) uniform sampler3D u_occupancy[3];
uniform vec3  u_occ_origins[3];
uniform float u_occ_inv_cells[3];
uniform int   u_occ_cascades;
uniform int   u_occ_mips;
#include "visibility.glsl"
#include "ltc.glsl"

in vec2 v_uv;
out vec4 frag_color;

layout(binding = 0) uniform sampler2D u_albedo;
layout(binding = 1) uniform sampler2D u_normal;
layout(binding = 2) uniform sampler2D u_emissive;
layout(binding = 3) uniform sampler2D u_motion;
layout(binding = 4) uniform sampler2D u_depth;
layout(binding = 5) uniform sampler2D u_indirect;      // S7 output, full res
layout(binding = 6) uniform sampler2DShadow u_sun_shadow;

layout(std430, binding = 0) readonly buffer Emitters { Emitter emitters[]; };
uniform uint u_emitter_count;
uniform int  u_emitter_steps;

uniform int   u_view_mode;
uniform float u_exposure;
uniform mat4  u_inv_view_proj;
uniform vec3  u_scene_min;
uniform vec3  u_scene_extent;

uniform vec3  u_sun_dir;         // points TOWARD the sun
uniform vec3  u_sun_radiance;
uniform mat4  u_sun_view_proj;
uniform float u_shadow_texel;
uniform int   u_have_sun;
uniform float u_emissive_boost;
uniform vec3  u_sky;
uniform vec3  u_cam_pos;
uniform float u_indirect_gain;

float sun_visibility(vec3 p, vec3 n) {
    vec4 c = u_sun_view_proj * vec4(p + n * u_shadow_texel * 2.0, 1.0);
    vec3 uv = c.xyz / c.w * 0.5 + 0.5;
    if (any(lessThan(uv.xy, vec2(0.0))) || any(greaterThan(uv.xy, vec2(1.0))) || uv.z > 1.0)
        return 1.0;
    return texture(u_sun_shadow, uv);
}

// GGX specular, single sun. This is the ROUGH end of the S8 ladder: the
// mirror-to-glossy bands (SSR, parallax cubemap) need a lit HDR target and a
// history copy to march against, which the pipeline does not have yet, so they
// are still open. Everything here is analytic and needs neither.
float d_ggx(float ndh, float a) {
    float a2 = a * a;
    float d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / max(3.14159265 * d * d, 1e-7);
}
float v_smith(float ndv, float ndl, float a) {
    // Height-correlated Smith visibility (already divided by 4*ndv*ndl).
    float a2 = a * a;
    float gv = ndl * sqrt(ndv * ndv * (1.0 - a2) + a2);
    float gl = ndv * sqrt(ndl * ndl * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-7);
}
vec3 f_schlick(vec3 f0, float u) {
    return f0 + (vec3(1.0) - f0) * pow(1.0 - u, 5.0);
}

// ACES filmic tonemap (Narkowicz fit).
vec3 tonemap_aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec4  alb   = texture(u_albedo, v_uv);
    vec4  nrm   = texture(u_normal, v_uv);
    vec4  emi   = texture(u_emissive, v_uv);
    float depth = texture(u_depth, v_uv).r;

    vec3 color;
    switch (u_view_mode) {
        case 1:  color = nrm.xyz * 0.5 + 0.5; break;
        case 2:  color = vec3(nrm.w); break;                      // roughness
        case 3:  color = vec3(emi.w); break;                      // metallic
        case 4:  color = emi.rgb; break;
        case 5:  color = vec3(abs(texture(u_motion, v_uv).xy) * 40.0, 0.0); break;
        case 6: {                                                 // world position
            vec3 p = world_from_depth(depth, v_uv, u_inv_view_proj);
            color = alb.a > 0.0 ? (p - u_scene_min) / max(u_scene_extent, vec3(1e-6)) : vec3(0.0);
            break;
        }
        case 7:  color = vec3(pow(depth, 64.0)); break;
        case 8:  color = texture(u_indirect, v_uv).rgb; break;    // indirect only
        case 9: {                                                 // LIT
            if (alb.a <= 0.0) { color = u_sky; break; }
            vec3 N = normalize(nrm.xyz);
            vec3 P = world_from_depth(depth, v_uv, u_inv_view_proj);

            // Indirect: the surfel cache, gathered and upsampled by S7. The value
            // stored is irradiance/pi, so multiplying by albedo directly gives
            // outgoing radiance for a Lambertian surface.
            vec3 indirect = texture(u_indirect, v_uv).rgb * u_indirect_gain;

            // Direct: sun only, one shadow tap. The full emitter tiering, LTC
            // area lights and the PCSS ladder are M5.
            // Area lights, analytic. Same closed form the cache uses, but with
            // a longer cone march: a pixel shows the penumbra directly, where
            // the cache only ever sees it through a hemisphere integral.
            vec3 direct = vec3(0.0);
            vec3 spec = vec3(0.0);
            for (uint k = 0u; k < u_emitter_count; ++k) {
                Emitter e = emitters[k];
                vec3 contrib = emitter_irradiance(P, N, e) * u_emissive_boost;
                if (dot(contrib, vec3(1.0)) <= 1e-6) continue;
                vec3 to = emitter_center(e) - P;
                float d = length(to);
                if (d < 1e-6) continue;
                float spread = clamp(sqrt(emitter_area(e)) * 0.5 / d, 0.02, 1.0);
                direct += contrib * cone_visibility(P, N, to / d, d, spread, u_emitter_steps);
            }
            if (u_have_sun != 0) {
                float ndl = max(dot(N, u_sun_dir), 0.0);
                if (ndl > 0.0) {
                    float vis = sun_visibility(P, N);
                    direct += u_sun_radiance * ndl * vis;

                    // Sun specular. Metals take their F0 from the albedo and
                    // have no diffuse lobe; dielectrics get a flat 4%.
                    float metal = emi.w;
                    float rough = clamp(nrm.w, 0.03, 1.0);
                    float a = rough * rough;
                    vec3 V = normalize(u_cam_pos - P);
                    vec3 H = normalize(V + u_sun_dir);
                    float ndv = max(dot(N, V), 1e-4);
                    float ndh = max(dot(N, H), 0.0);
                    vec3 f0 = mix(vec3(0.04), alb.rgb, metal);
                    spec = d_ggx(ndh, a) * v_smith(ndv, ndl, a)
                         * f_schlick(f0, max(dot(u_sun_dir, H), 0.0))
                         * u_sun_radiance * ndl * vis;
                }
            }

            float kd = 1.0 - emi.w;   // metals have no diffuse lobe
            color = emi.rgb * u_emissive_boost
                  + alb.rgb * kd * (direct + indirect)
                  + spec;
            break;
        }
        default: color = alb.rgb; break;                          // 0: albedo
    }

    // Debug views are already display-referred; only the lit paths are tonemapped.
    if (u_view_mode == 0 || u_view_mode == 4 || u_view_mode == 8 || u_view_mode == 9) {
        color = tonemap_aces(color * u_exposure);
        color = pow(color, vec3(1.0 / 2.2));
    }
    frag_color = vec4(color, 1.0);
}
