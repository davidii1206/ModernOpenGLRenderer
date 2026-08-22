#version 460 core
in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_tex;
uniform int u_mode;      // 0=position 1=normal 2=albedo 3=emissive 4=depth
uniform float u_far;
uniform float u_exposure;
uniform float u_gamma;

vec3 aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec4 c = texture(u_tex, v_uv);

    if (u_mode == 0) {
        c = vec4(c.rgb * 0.5 + 0.5, 1.0);          // position -> remap
    } else if (u_mode == 1) {
        c = vec4(c.rgb * 0.5 + 0.5, 1.0);          // normal  -> remap
    } else if (u_mode == 2) {
        c = vec4(c.rgb, 1.0);                       // albedo  -> pass
    } else if (u_mode == 3) {
        c = vec4(c.rgb, 1.0);                       // emissive-> pass
    } else if (u_mode == 4) {
        c = vec4(vec3(c.r / u_far), 1.0);           // depth   -> linear
    }

    vec3 col = c.rgb * u_exposure;
    col = aces(col);
    col = pow(max(col, vec3(0.0)), vec3(1.0 / u_gamma));
    frag_color = vec4(col, 1.0);
}
