#version 460 core

in vec3 v_normal;
in vec3 v_color;
in vec2 v_motion;

layout (location = 0) out vec4 frag_color;
layout (location = 1) out vec2 out_motion;

void main() {
    vec3 n = normalize(v_normal);
    float l = max(dot(n, normalize(vec3(1, 2, 3))), 0.0);
    frag_color = vec4(v_color * (0.3 + 0.7 * l), 1.0);
    out_motion = v_motion;
}
