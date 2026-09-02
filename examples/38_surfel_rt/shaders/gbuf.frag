#version 460 core
in vec3 v_pos;
in vec3 v_normal;

layout(location = 0) out vec4 out_direct;  // RGBA16F: emissive term only
layout(location = 1) out vec4 out_pos;     // RGBA16F: world position
layout(location = 2) out vec4 out_nrm;     // RGBA16F: world normal
layout(location = 3) out vec4 out_alb;     // RGBA8:  albedo.rgb + mirror flag.a

uniform vec3 u_albedo;
uniform float u_is_mirror;
uniform vec3 u_emissive;   // emissive factor x strength (0 for non-emitters)

void main() {
    vec3 n = normalize(v_normal);

    out_pos = vec4(v_pos, 1.0);
    out_nrm = vec4(n, 0.0);
    out_alb = vec4(u_albedo, u_is_mirror);

    // The raster pass stores only the EMISSIVE term: all illumination comes
    // from the cone gather over the emissive surfels (radiance cache). The
    // light patch displays itself; everything else is black until lit by the
    // gather composite.
    out_direct = vec4(u_emissive, 1.0);
}
