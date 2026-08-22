#version 460 core
layout(std430, binding = 0) readonly buffer PGeomF { vec4 pgeom[]; };
layout(std430, binding = 1) readonly buffer PNrmF  { vec4 pnrm[]; };
layout(std430, binding = 2) readonly buffer PAlbF  { vec4 palb[]; };
layout(std430, binding = 3) readonly buffer PEmitF { vec4 pemit[]; };
layout(std430, binding = 4) readonly buffer PLeafP { uvec4 leaf_packed[]; };
layout(std430, binding = 8) readonly buffer RadF   { vec4 rad[]; };

uniform mat4 u_view_proj;
uniform float u_point_size;
uniform int u_color_mode;   // 0=albedo 1=emissive 2=normal 3=position 4=radiance
uniform uint u_num_leaves;
uniform uint u_rad_offset;  // interior node count; leaf nodes start here
uniform int u_use_packed;
uniform vec3 u_scene_min;
uniform vec3 u_scene_size;
uniform float u_radius_scale;
uniform float u_rad_gain;

out vec3 v_color;

vec3 oct_decode(vec2 p) {
    vec3 n = vec3(p, 1.0 - abs(p.x) - abs(p.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(vec2(n.y, n.x))) * sign(n.xy);
    return normalize(n);
}

void main() {
    uint i = gl_VertexID;
    vec3 pos;
    vec3 nrm;
    vec3 alb;
    vec3 emi;

    if (u_use_packed != 0) {
        uvec4 L = leaf_packed[i];
        uint px = L.x >> 16u, r = L.x & 0xFFFFu;
        uint py = L.y >> 16u, pz = L.y & 0xFFFFu;
        pos = u_scene_min + u_scene_size * (vec3(float(px), float(py), float(pz)) / 65535.0);
        int hi = int(L.w >> 16u);  hi = hi > 32767 ? hi - 65536 : hi;
        int lo = int(L.w & 0xFFFFu); lo = lo > 32767 ? lo - 65536 : lo;
        nrm = oct_decode(vec2(float(hi), float(lo)) / 32767.0);
        alb = vec3(float((L.z >> 16u) & 0xFFu),
                   float((L.z >> 8u)  & 0xFFu),
                   float(L.z & 0xFFu)) / 255.0;
        emi = vec3(0.0);
    } else {
        pos = pgeom[i].xyz;
        nrm = pnrm[i].xyz;
        alb = palb[i].rgb;
        emi = pemit[i].rgb;
    }

    gl_Position = u_view_proj * vec4(pos, 1.0);
    gl_PointSize = u_point_size;

    if (u_color_mode == 0)      v_color = alb;
    else if (u_color_mode == 1) v_color = emi;
    else if (u_color_mode == 2) v_color = nrm * 0.5 + 0.5;
    else if (u_color_mode == 3) v_color = pos * 0.5 + 0.5;
    else                        v_color = rad[u_rad_offset + i].rgb * u_rad_gain;
}
