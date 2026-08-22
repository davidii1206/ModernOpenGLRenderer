#version 460 core
in vec3 v_pos;
in vec3 v_normal;

layout(location = 0) out vec4 out_direct;  // RGBA16F: direct point-light shaded color
layout(location = 1) out vec4 out_pos;     // RGBA16F: world position
layout(location = 2) out vec4 out_nrm;     // RGBA16F: world normal
layout(location = 3) out vec4 out_alb;     // RGBA8:  albedo.rgb + mirror flag.a

uniform vec3 u_albedo;
uniform vec3 u_light_pos;
uniform vec3 u_light_color;
uniform float u_light_intensity;
uniform float u_ambient;
uniform float u_is_mirror;

void main() {
    vec3 n = normalize(v_normal);

    out_pos = vec4(v_pos, 1.0);
    out_nrm = vec4(n, 0.0);
    out_alb = vec4(u_albedo, u_is_mirror);

    // Direct point-light shading (no shadows). Mirror faces have ~zero albedo,
    // so their direct term is negligible; they get their color from reflections.
    vec3 to_light = u_light_pos - v_pos;
    float dist2 = dot(to_light, to_light);
    float dist = sqrt(dist2);
    vec3 L = to_light / max(dist, 1e-5);
    float ndl = max(dot(n, L), 0.0);
    vec3 direct = u_albedo * (u_light_color * u_light_intensity) * ndl / max(dist2, 1e-4)
                + u_albedo * u_ambient;
    out_direct = vec4(direct, 1.0);
}