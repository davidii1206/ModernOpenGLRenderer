#version 460 core
layout(location = 0) in vec3 a_pos;    // unit sphere vertex
layout(location = 1) in vec4 a_sphere; // center.xyz, radius (instanced)

uniform mat4 u_view_proj;

void main() {
    gl_Position = u_view_proj * vec4(a_sphere.xyz + a_pos * a_sphere.w, 1.0);
}
