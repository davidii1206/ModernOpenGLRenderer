#version 460 core
in vec2 v_uv;
out vec4 frag_color;

// The trace kernel only writes pixels that cast rays (mirror surfaces); the
// composite finishes here: mirror pixels take the traced color, everything
// else the rasterized direct shading. Debug views read the rasterized
// G-buffer textures directly (the old aux image was a redundant copy of them).
uniform sampler2D u_tex;      // traced color (view 0) or the debug source texture
uniform sampler2D u_direct;   // rasterized direct shading
uniform sampler2D u_mirror;   // rasterized albedo (alpha = mirror flag)
uniform int u_view;           // 0 composite, 1 albedo, 2 normal, 3 position, 4 depth
uniform int u_composite;      // 1 = blend u_tex over u_direct by the mirror flag
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
        col = texture(u_direct, v_uv).rgb;
        if (u_composite == 1) {
            // The accumulation image is fp32; the old single-dispatch pipeline
            // stored its total through an RGBA16F texture before display, so
            // round to half float here to stay byte-identical.
            vec3 rt = texture(u_tex, v_uv).rgb;
            float m = texture(u_mirror, v_uv).a;
            col = mix(col, rt, step(0.5, m));
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
