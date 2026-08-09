#version 460 core
#extension GL_ARB_shader_draw_parameters : enable

layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec3 a_normal;

struct InstanceData {
    mat4 model;
    vec4 bounding_sphere;
    uint model_id;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

layout(std430, binding=0) buffer Instances {
    InstanceData data[];
} instances;

layout(std430, binding=2) buffer VisibleIDs {
    uint ids[];
} visible_ids;

uniform mat4 u_view_proj;
uniform mat4 u_prev_view_proj;
uniform vec2 u_render_size;
uniform uint u_max_instances;
uniform int  u_use_multi_draw;

uniform uint u_draw_lod_level[64];
uniform int  u_total_slots;

uniform uint u_lod_offset;
uniform int  u_lod_level;

out vec3 v_normal;
out vec3 v_color;
out vec2 v_motion;

const vec3 lod_colors[7] = vec3[](
    vec3(0.2, 0.6, 1.0),
    vec3(0.2, 1.0, 0.4),
    vec3(1.0, 0.8, 0.2),
    vec3(1.0, 0.4, 0.2),
    vec3(1.0, 0.2, 0.2),
    vec3(0.8, 0.2, 1.0),
    vec3(0.5, 0.5, 0.5)
);

void main() {
    int lod;
    uint offset;
    if (u_use_multi_draw != 0) {
        int slot = gl_DrawID;
        lod = int(u_draw_lod_level[slot]);
        offset = uint(slot) * u_max_instances;
    } else {
        offset = u_lod_offset;
        lod = u_lod_level;
    }
    uint idx = visible_ids.ids[offset + gl_InstanceID];
    mat4 m = instances.data[idx].model;
    vec4 curr_pos = u_view_proj * m * vec4(a_pos, 1.0);
    gl_Position = curr_pos;
    v_normal = mat3(m) * a_normal;
    v_color = lod_colors[lod];
    vec4 prev_pos = u_prev_view_proj * m * vec4(a_pos, 1.0);
    vec2 curr_ndc = curr_pos.xy / curr_pos.w;
    vec2 prev_ndc = prev_pos.xy / prev_pos.w;
    v_motion = (curr_ndc - prev_ndc) * u_render_size * 0.5;
}
