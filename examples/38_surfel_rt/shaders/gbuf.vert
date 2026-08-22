#version 460 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;

uniform mat4 u_view_proj;
uniform mat4 u_model;
uniform mat3 u_normal_mat;

out vec3 v_pos;
out vec3 v_normal;

void main() {
    vec4 world = u_model * vec4(a_pos, 1.0);
    gl_Position = u_view_proj * world;
    v_pos = world.xyz;
    v_normal = normalize(u_normal_mat * a_normal);
}