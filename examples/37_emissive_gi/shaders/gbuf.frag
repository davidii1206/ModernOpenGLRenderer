#version 460 core
in vec3 v_world_pos;
in vec3 v_world_normal;
in vec2 v_uv;

layout(location = 0) out vec4 out_position;
layout(location = 1) out vec4 out_normal;
layout(location = 2) out vec4 out_albedo;
layout(location = 3) out vec4 out_emissive;
layout(location = 4) out float out_depth;

uniform vec3 u_albedo;
uniform vec3 u_emissive;
uniform mat4 u_view;

uniform bool u_has_base_tex;
uniform bool u_has_emissive_tex;
uniform sampler2D u_base_tex;
uniform sampler2D u_emissive_tex;

void main() {
    vec3 n = normalize(v_world_normal);
    vec4 view_p = u_view * vec4(v_world_pos, 1.0);

    vec3 albedo = u_has_base_tex ? texture(u_base_tex, v_uv).rgb : u_albedo;
    vec3 emissive = u_has_emissive_tex ? texture(u_emissive_tex, v_uv).rgb : u_emissive;

    out_position = vec4(v_world_pos, 1.0);
    out_normal   = vec4(n, 1.0);
    out_albedo   = vec4(albedo, 1.0);
    out_emissive = vec4(emissive, 1.0);
    out_depth    = -view_p.z;
}
