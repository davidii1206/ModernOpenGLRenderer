#version 460 core
layout(std430, binding = 0) readonly buffer SurfPos { vec4 surfel_pos[]; };
layout(std430, binding = 1) readonly buffer SurfNrm { vec4 surfel_nrm[]; };
layout(std430, binding = 2) readonly buffer SurfAlb { vec4 surfel_alb[]; };

uniform mat4 u_view_proj;
uniform float u_point_size;
uniform int u_color_mode;   // 0=albedo 1=normal 2=position 3=mirror-flag
uniform uint u_num_surfels;

out vec3 v_color;

void main() {
    uint i = gl_VertexID;
    if (i >= u_num_surfels) {
        gl_Position = vec4(0.0);
        gl_PointSize = 1.0;
        v_color = vec3(0.0);
        return;
    }
    vec3 pos = surfel_pos[i].xyz;
    vec3 nrm = surfel_nrm[i].xyz;
    vec3 alb = surfel_alb[i].rgb;

    gl_Position = u_view_proj * vec4(pos, 1.0);
    gl_PointSize = u_point_size;

    if (u_color_mode == 0)      v_color = alb;
    else if (u_color_mode == 1) v_color = nrm * 0.5 + 0.5;
    else if (u_color_mode == 2) v_color = pos * 0.5 + 0.5;
    else                        v_color = vec3(surfel_alb[i].w);
}