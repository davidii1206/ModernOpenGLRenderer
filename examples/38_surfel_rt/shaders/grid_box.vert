#version 460 core
layout(location = 0) in vec3 a_pos;     // unit cube vertex [-1,1]^3 (scaled to cell)
layout(location = 1) in vec4 a_cell;    // cell min.xyz + cell size.w (instanced)

uniform mat4 u_view_proj;

void main() {
    vec3 center = a_cell.xyz + a_cell.w * 0.5;
    vec3 hsize = vec3(a_cell.w) * 0.5;
    gl_Position = u_view_proj * vec4(center + a_pos * hsize, 1.0);
}