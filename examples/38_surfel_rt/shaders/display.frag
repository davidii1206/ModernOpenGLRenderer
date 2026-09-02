#version 460 core
in vec2 v_uv;
out vec4 frag_color;

// Composite: non-mirror pixels = emissive term + albedo x cone-gathered
// irradiance (bilinear-upsampled from the half-res GI buffer); mirror pixels
// take the ray-traced radiance. Debug views read the G-buffer directly.
uniform sampler2D u_tex;      // traced radiance (view 0) or the debug source texture
uniform sampler2D u_direct;   // emissive term per pixel
uniform sampler2D u_mirror;   // rasterized albedo (alpha = mirror flag)
uniform sampler2D u_gi;       // half-res gather irradiance (rgb) + ao (a)
uniform int u_view;           // 0 composite, 1 albedo, 2 normal, 3 position, 4 depth
uniform int u_composite;      // 1 = mirror pixels take u_tex
uniform vec3 u_cam_pos;
uniform float u_far;
uniform float u_exposure;
uniform float u_gamma;

vec3 aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 col;
    if (u_view == 0) {
        vec4 alb_m = texture(u_mirror, v_uv);
        // The accumulated radiance is fp32; round to half float like the
        // legacy path so the mirror content stays stable across refactors.
        vec3 rt = texture(u_tex, v_uv).rgb;
        vec2 h0 = unpackHalf2x16(packHalf2x16(rt.xy));
        vec2 h1 = unpackHalf2x16(packHalf2x16(rt.zz));
        vec3 rt16 = vec3(h0.x, h0.y, h1.x);
        if (u_composite == 1 && step(0.5, alb_m.a) == 1.0) {
            col = rt16;
        } else {
            // Diffuse: emissive term + albedo x gathered irradiance.
            col = texture(u_direct, v_uv).rgb
                + alb_m.rgb * texture(u_gi, v_uv).rgb;
        }
    } else {
        vec4 g = texture(u_tex, v_uv);
        if (u_view == 1)      col = g.rgb;
        else if (u_view == 2) col = g.rgb * 0.5 + 0.5;
        else if (u_view == 3) col = g.rgb * 0.5 + 0.5;
        else                  col = vec3(length(g.rgb - u_cam_pos) / u_far);
    }
    col *= u_exposure;
    col = aces(col);
    col = pow(max(col, vec3(0.0)), vec3(1.0 / u_gamma));
    frag_color = vec4(col, 1.0);
}
