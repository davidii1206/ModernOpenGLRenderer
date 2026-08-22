#version 460 core
in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_tex;
uniform float u_exposure;
uniform float u_gamma;

vec3 aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 col = texture(u_tex, v_uv).rgb * u_exposure;
    col = aces(col);
    col = pow(max(col, vec3(0.0)), vec3(1.0 / u_gamma));
    frag_color = vec4(col, 1.0);
}