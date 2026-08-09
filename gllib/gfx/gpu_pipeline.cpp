#include "gpu_pipeline.hpp"

#include <gl/shader.hpp>
#include <gfx/mesh.hpp>
#include <gfx/camera.hpp>
#include <gfx/model.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <gllib/log.hpp>

#include <cstdio>
#include <cstring>

namespace gfx {

// Pass 1: frustum cull + Hi-Z cull + LOD selection + pack visible instance IDs
static const char* cull_comp_src = R"(
#version 460 core
#define MAX_LODS 8
#define MAX_GROUPS 8
#define MAX_SLOTS (MAX_GROUPS * MAX_LODS)
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct InstanceData {
    mat4 model;
    vec4 bounding_sphere;
    uint model_id;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

layout(std430, binding=0) readonly buffer Instances {
    InstanceData data[];
} instances;

layout(std430, binding=2) buffer VisibleIDs {
    uint ids[];
} visible_ids;

layout(std430, binding=3) buffer GroupCounts {
    uint counts[];
} group_counts;

uniform uint u_instance_count;
uniform uint u_max_instances;
uniform vec4 u_frustum_planes[6];
uniform mat4 u_view_proj;

// Hi-Z
layout(binding=4) uniform sampler2D u_hiz;
uniform vec2  u_screen_size;
uniform float u_proj_00;
uniform float u_proj_11;
uniform int   u_hiz_max_level;

// LOD thresholds
uniform float u_lod_thresholds[7];
uniform int   u_group_lod_counts[MAX_GROUPS];
uniform int   u_group_count;

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= u_instance_count) return;

    InstanceData inst = instances.data[id];
    uint gid = inst.model_id;
    if (gid >= uint(u_group_count)) return;

    vec4 center_w = inst.model * vec4(inst.bounding_sphere.xyz, 1.0);
    float scale   = length(inst.model[0].xyz);
    float radius  = inst.bounding_sphere.w * scale;

    // Frustum cull
    for (int i = 0; i < 6; ++i) {
        float d = dot(u_frustum_planes[i].xyz, center_w.xyz) + u_frustum_planes[i].w;
        if (d <= -radius) return;
    }

    // Project sphere center to clip space
    vec4 clip = u_view_proj * vec4(center_w.xyz, 1.0);

    // Hi-Z occlusion cull
    if (u_hiz_max_level > 0) {
        float proj_half_x = radius * u_proj_00 / abs(clip.w);
        float proj_half_y = radius * u_proj_11 / abs(clip.w);

        vec2 ndc = clip.xy / clip.w;
        vec2 tc = ndc * 0.5 + 0.5;

        vec2 half_uv = vec2(proj_half_x, proj_half_y) * 0.5;
        vec2 tc_min = max(tc - half_uv, vec2(0.0));
        vec2 tc_max = min(tc + half_uv, vec2(1.0));
        vec2 tc_size = tc_max - tc_min;

        vec2 pixel_size2 = tc_size * u_screen_size;
        float max_pixels = max(pixel_size2.x, pixel_size2.y);
        int mip = clamp(int(ceil(log2(max_pixels))), 0, u_hiz_max_level);

        vec2 center_tc = (tc_min + tc_max) * 0.5;
        float z_max = textureLod(u_hiz, center_tc, mip).r;
        float depth = clip.z / clip.w * 0.5 + 0.5;
        if (depth > z_max + 0.0005) return;
    }

    // LOD selection
    float proj_half = max(radius * u_proj_00, radius * u_proj_11) / abs(clip.w);
    float pixel_size = proj_half * u_screen_size.y;
    int lod = 0;
    int group_lod_count = u_group_lod_counts[gid];
    for (int i = 0; i < 7 && i < group_lod_count - 1; ++i) {
        if (pixel_size < u_lod_thresholds[i]) lod = i + 1;
    }

    // Pack into per-(group, lod) visible array
    uint slot = gid * MAX_LODS + uint(lod);
    uint idx = atomicAdd(group_counts.counts[slot], 1u);
    visible_ids.ids[slot * u_max_instances + idx] = id;
}
)";

// Pass 2: write per-slot indirect commands from packed counts
static const char* pack_comp_src = R"(
#version 460 core
#define MAX_LODS 8
#define MAX_GROUPS 8
#define MAX_SLOTS (MAX_GROUPS * MAX_LODS)
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct DrawCommand {
    uint count;
    uint instanceCount;
    uint firstIndex;
    int  baseVertex;
    uint baseInstance;
};

layout(std430, binding=1) buffer IndirectCmds {
    DrawCommand cmds[];
} indirect;

layout(std430, binding=3) buffer GroupCounts {
    uint counts[];
} group_counts;

uniform uint u_slot_index_count[MAX_SLOTS];
uniform uint u_slot_first_index[MAX_SLOTS];
uniform int  u_slot_base_vertex[MAX_SLOTS];
uniform int  u_total_slots;

void main() {
    for (int slot = 0; slot < MAX_SLOTS && slot < u_total_slots; ++slot) {
        uint visible = group_counts.counts[slot];
        indirect.cmds[slot].count         = u_slot_index_count[slot];
        indirect.cmds[slot].instanceCount = visible;
        indirect.cmds[slot].firstIndex    = u_slot_first_index[slot];
        indirect.cmds[slot].baseVertex    = u_slot_base_vertex[slot];
        indirect.cmds[slot].baseInstance  = 0;
        group_counts.counts[slot] = 0;
    }
}
)";

static void extract_frustum_planes(const glm::mat4& vp, glm::vec4* planes) {
    glm::mat4 t = glm::transpose(vp);
    planes[0] = t[3] + t[0];
    planes[1] = t[3] - t[0];
    planes[2] = t[3] + t[1];
    planes[3] = t[3] - t[1];
    planes[4] = t[3] + t[2];
    planes[5] = t[3] - t[2];
    for (int i = 0; i < 6; ++i) {
        float len = glm::length(glm::vec3(planes[i]));
        if (len > 0.0f) planes[i] /= len;
    }
}

static GLuint compile_compute(const char* src, const char* name) {
    gl::Shader cs(gl::ShaderType::compute, src);
    if (!cs.compiled()) return 0;
    GLuint prog = glCreateProgram();
    glAttachShader(prog, cs.handle());
    glLinkProgram(prog);
    GLint status = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &status);
    if (!status) {
        char buf[512];
        GLsizei len = 0;
        glGetProgramInfoLog(prog, sizeof(buf), &len, buf);
        gllib::logf(gllib::LogLevel::error, "GpuPipeline %s link: %s", name, buf);
        glDeleteProgram(prog);
        return 0;
    }
    return prog;
}

// --- Constructor / Destructor ---

GpuPipeline::GpuPipeline(int max_instances)
    : max_instances_(max_instances)
    , instances_(static_cast<size_t>(max_instances))
{
}

GpuPipeline::~GpuPipeline() {
    if (instance_ssbo_) glDeleteBuffers(1, &instance_ssbo_);
    if (indirect_ssbo_) glDeleteBuffers(1, &indirect_ssbo_);
    if (visible_ssbo_)  glDeleteBuffers(1, &visible_ssbo_);
    if (counter_ssbo_)  glDeleteBuffers(1, &counter_ssbo_);
    if (draw_count_ssbo_) glDeleteBuffers(1, &draw_count_ssbo_);
    if (merged_vao_) glDeleteVertexArrays(1, &merged_vao_);
    if (merged_vbo_) glDeleteBuffers(1, &merged_vbo_);
    if (merged_ebo_) glDeleteBuffers(1, &merged_ebo_);
    if (cull_prog_)     glDeleteProgram(cull_prog_);
    if (pack_prog_)     glDeleteProgram(pack_prog_);
}

// --- Registration ---

int GpuPipeline::add_model(const Model& model, int lod_group_index) {
    if (finalized_) {
        gllib::logf(gllib::LogLevel::error,
            "GpuPipeline: add_model() called after finalize()");
        return -1;
    }

    int gid = static_cast<int>(registered_models_.size());
    if (gid >= kMaxGroups) {
        gllib::logf(gllib::LogLevel::error,
            "GpuPipeline: too many models (max %d)", kMaxGroups);
        return -1;
    }

    const auto& group = model.lod_group(lod_group_index);
    int cnt = static_cast<int>(group.mesh_indices.size());
    if (cnt > kMaxLods) cnt = kMaxLods;
    group_lod_counts_[gid] = cnt;
    if (cnt > max_lod_count_) max_lod_count_ = cnt;

    // Set up per-slot info (indices populated during build_merged_vao)
    for (int l = 0; l < cnt; ++l) {
        int s = slot_for(gid, l);
        slot_lod_level_[s] = static_cast<GLuint>(l);
    }

    ModelEntry entry;
    entry.model = &model;
    entry.lod_group_index = lod_group_index;
    entry.group_id = gid;
    const auto& lg = model.lod_group(lod_group_index);
    entry.local_bs = model.mesh_bounding_sphere(lg.mesh_indices[0]);
    registered_models_.push_back(entry);

    group_count_ = gid + 1;
    total_slots_ = group_count_ * kMaxLods;

    gllib::logf(gllib::LogLevel::debug, "GpuPipeline: added model %d, %d LODs, %d slots",
        gid, cnt, total_slots_);
    return gid;
}

void GpuPipeline::finalize() {
    if (finalized_) return;

    build_merged_vao();
    init_ssbo();
    init_shaders();
    init_indirect_params();
    finalized_ = true;

    gllib::logf(gllib::LogLevel::info,
        "GpuPipeline: finalized with %d models, %d LOD groups, %d slots",
        group_count_, max_lod_count_, total_slots_);
}

// --- Submission ---

void GpuPipeline::draw(int model_handle, const glm::mat4& transform) {
    if (model_handle < 0 || model_handle >= static_cast<int>(registered_models_.size()))
        return;
    const auto& entry = registered_models_[model_handle];

    InstanceData inst;
    inst.model = transform;
    inst.model_id = static_cast<GLuint>(entry.group_id);
    inst.bounding_sphere = entry.local_bs; // local-space — cull shader applies model transform

    pending_draws_.push_back(inst);
}

void GpuPipeline::clear() {
    pending_draws_.clear();
}

void GpuPipeline::flush(const Camera& camera, GLuint render_program) {
    if (!finalized_) {
        gllib::logf(gllib::LogLevel::error,
            "GpuPipeline: flush() called before finalize()");
        return;
    }

    int count = static_cast<int>(pending_draws_.size());
    if (count == 0) return;
    if (count > max_instances_) count = max_instances_;

    for (int i = 0; i < count; ++i)
        instances_[i] = pending_draws_[i];
    pending_draws_.clear();

    active_instances_ = count;
    upload();
    cull_and_draw(camera, render_program);
}

// --- SSBOs ---

void GpuPipeline::init_ssbo() {
    glCreateBuffers(1, &instance_ssbo_);
    glNamedBufferStorage(instance_ssbo_,
        max_instances_ * sizeof(InstanceData),
        instances_.data(),
        GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, instance_ssbo_);

    // Indirect command buffer — kMaxSlots × DrawElementsIndirectCommand
    glCreateBuffers(1, &indirect_ssbo_);
    GLuint init_cmd[5 * kMaxSlots] = {};
    glNamedBufferStorage(indirect_ssbo_, sizeof(init_cmd), init_cmd,
        GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, indirect_ssbo_);

    // Per-slot visible instance IDs
    glCreateBuffers(1, &visible_ssbo_);
    glNamedBufferStorage(visible_ssbo_,
        kMaxSlots * max_instances_ * sizeof(GLuint),
        nullptr,
        GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, visible_ssbo_);

    // Per-slot atomic counters
    glCreateBuffers(1, &counter_ssbo_);
    GLuint zero[kMaxSlots] = {};
    glNamedBufferStorage(counter_ssbo_, sizeof(zero), zero,
        GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, counter_ssbo_);
}

// --- Shaders ---

void GpuPipeline::init_shaders() {
    cull_prog_ = compile_compute(cull_comp_src, "cull");
    pack_prog_ = compile_compute(pack_comp_src, "pack");
}

void GpuPipeline::init_indirect_params() {
    GLint ext_count = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &ext_count);
    for (int i = 0; i < ext_count; ++i) {
        const char* ext = reinterpret_cast<const char*>(glGetStringi(GL_EXTENSIONS, i));
        if (std::strcmp(ext, "GL_ARB_indirect_parameters") == 0) {
            has_indirect_params_ = true;
        }
    }

    if (has_indirect_params_) {
        glCreateBuffers(1, &draw_count_ssbo_);
        GLint draw_count = static_cast<GLint>(total_slots_);
        glNamedBufferStorage(draw_count_ssbo_, sizeof(draw_count),
            &draw_count, GL_DYNAMIC_STORAGE_BIT);
        gllib::logf(gllib::LogLevel::info,
            "GpuPipeline: GL_ARB_indirect_parameters available — single multi-draw");
    } else {
        gllib::logf(gllib::LogLevel::info,
            "GpuPipeline: GL_ARB_indirect_parameters not available — per-LOD draws");
    }
}

// --- Merged VAO ---

void GpuPipeline::build_merged_vao() {
    if (registered_models_.empty()) return;

    const Mesh* mesh_ptrs[kMaxSlots] = {};
    for (int g = 0; g < group_count_; ++g) {
        const auto& entry = registered_models_[g];
        const auto& group = entry.model->lod_group(entry.lod_group_index);
        int cnt = std::min(static_cast<int>(group.mesh_indices.size()), kMaxLods);
        for (int l = 0; l < cnt; ++l) {
            int s = slot_for(g, l);
            mesh_ptrs[s] = &entry.model->mesh(group.mesh_indices[l]);
        }
    }

    // Compute total vertices and indices
    GLuint total_verts = 0, total_indices = 0;
    GLuint vert_offsets[kMaxSlots] = {};
    GLuint idx_offsets[kMaxSlots] = {};

    for (int s = 0; s < kMaxSlots; ++s) {
        if (!mesh_ptrs[s]) continue;
        auto vc = static_cast<GLuint>(mesh_ptrs[s]->vertex_count());
        auto ic = static_cast<GLuint>(mesh_ptrs[s]->index_count());
        vert_offsets[s] = total_verts;
        idx_offsets[s]  = total_indices;
        total_verts += vc;
        total_indices += ic;
    }

    // Merged VBO
    glCreateBuffers(1, &merged_vbo_);
    glNamedBufferStorage(merged_vbo_, total_verts * sizeof(Vertex),
        nullptr, GL_DYNAMIC_STORAGE_BIT);

    for (int s = 0; s < kMaxSlots; ++s) {
        if (!mesh_ptrs[s]) continue;
        auto vc = static_cast<size_t>(mesh_ptrs[s]->vertex_count());
        if (vc == 0) continue;
        glCopyNamedBufferSubData(
            mesh_ptrs[s]->vbo_handle(), merged_vbo_,
            0, vert_offsets[s] * sizeof(Vertex),
            vc * sizeof(Vertex));
    }

    // Merged EBO
    glCreateBuffers(1, &merged_ebo_);
    glNamedBufferStorage(merged_ebo_, total_indices * sizeof(GLuint),
        nullptr, GL_DYNAMIC_STORAGE_BIT);

    for (int s = 0; s < kMaxSlots; ++s) {
        if (!mesh_ptrs[s]) continue;
        auto ic = static_cast<size_t>(mesh_ptrs[s]->index_count());
        if (ic == 0) continue;
        glCopyNamedBufferSubData(
            mesh_ptrs[s]->ebo_handle(), merged_ebo_,
            0, idx_offsets[s] * sizeof(GLuint),
            ic * sizeof(GLuint));
    }

    // VAO
    glCreateVertexArrays(1, &merged_vao_);
    glVertexArrayVertexBuffer(merged_vao_, 0, merged_vbo_, 0, sizeof(Vertex));

    glVertexArrayAttribFormat(merged_vao_, 0, 3, GL_FLOAT, GL_FALSE, offsetof(Vertex, position));
    glVertexArrayAttribBinding(merged_vao_, 0, 0);
    glEnableVertexArrayAttrib(merged_vao_, 0);

    glVertexArrayAttribFormat(merged_vao_, 1, 3, GL_FLOAT, GL_FALSE, offsetof(Vertex, normal));
    glVertexArrayAttribBinding(merged_vao_, 1, 0);
    glEnableVertexArrayAttrib(merged_vao_, 1);

    glVertexArrayAttribFormat(merged_vao_, 2, 2, GL_FLOAT, GL_FALSE, offsetof(Vertex, texcoord));
    glVertexArrayAttribBinding(merged_vao_, 2, 0);
    glEnableVertexArrayAttrib(merged_vao_, 2);

    glVertexArrayAttribFormat(merged_vao_, 3, 4, GL_FLOAT, GL_FALSE, offsetof(Vertex, tangent));
    glVertexArrayAttribBinding(merged_vao_, 3, 0);
    glEnableVertexArrayAttrib(merged_vao_, 3);

    glVertexArrayElementBuffer(merged_vao_, merged_ebo_);

    // Update per-slot draw info with merged offsets
    for (int s = 0; s < kMaxSlots; ++s) {
        if (!mesh_ptrs[s]) continue;
        auto ic = static_cast<GLuint>(mesh_ptrs[s]->index_count());
        if (ic == 0) continue;
        slot_first_index_[s] = idx_offsets[s];
        slot_base_vertex_[s] = static_cast<GLint>(vert_offsets[s]);
        slot_index_count_[s] = ic;
    }
}

// --- Upload ---

void GpuPipeline::upload() {
    glNamedBufferSubData(instance_ssbo_, 0,
        active_instances_ * sizeof(InstanceData),
        instances_.data());
}

// --- Hi-Z ---

void GpuPipeline::set_hiz(GLuint texture, float proj_00, float proj_11,
                          int screen_w, int screen_h, int max_level)
{
    hiz_texture_ = texture;
    hiz_proj_00_ = proj_00;
    hiz_proj_11_ = proj_11;
    hiz_screen_w_ = screen_w;
    hiz_screen_h_ = screen_h;
    hiz_max_level_ = max_level;
    hiz_active_ = (texture != 0 && max_level > 0);
}

// --- LOD thresholds ---

void GpuPipeline::set_lod_thresholds(const float* thresholds, int count) {
    int n = (count > kMaxLods - 1) ? (kMaxLods - 1) : count;
    for (int i = 0; i < n; ++i) lod_thresholds_[i] = thresholds[i];
    lod_threshold_count_ = n;
}

// --- Cull + Draw (internal) ---

void GpuPipeline::cull_and_draw(const Camera& camera, GLuint render_program) {
    if (!cull_prog_ || !pack_prog_) {
        std::fprintf(stderr, "GpuPipeline: null handles\n");
        return;
    }

    // Collect last frame's timer results
    if (timer_started_) {
        if (cull_timer_.result_available()) last_cull_time_ns_ = cull_timer_.result();
        if (pack_timer_.result_available()) last_pack_time_ns_ = pack_timer_.result();
        if (draw_timer_.result_available()) last_draw_time_ns_ = draw_timer_.result();
        last_total_time_ns_ = last_cull_time_ns_ + last_pack_time_ns_ + last_draw_time_ns_;
    }

    view_proj_ = camera.view_projection();

    // --- Pass 1: Cull + LOD ---
    cull_timer_.begin();
    glUseProgram(cull_prog_);

    glUniform1ui(glGetUniformLocation(cull_prog_, "u_instance_count"),
        static_cast<GLuint>(active_instances_));
    glUniform1ui(glGetUniformLocation(cull_prog_, "u_max_instances"),
        static_cast<GLuint>(max_instances_));

    glm::vec4 planes[6];
    extract_frustum_planes(view_proj_, planes);
    glUniform4fv(glGetUniformLocation(cull_prog_, "u_frustum_planes[0]"), 6, glm::value_ptr(planes[0]));

    glUniformMatrix4fv(glGetUniformLocation(cull_prog_, "u_view_proj"),
        1, GL_FALSE, glm::value_ptr(view_proj_));

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, instance_ssbo_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, visible_ssbo_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, counter_ssbo_);

    if (hiz_active_) {
        glUniform2f(glGetUniformLocation(cull_prog_, "u_screen_size"),
            static_cast<float>(hiz_screen_w_), static_cast<float>(hiz_screen_h_));
        glUniform1f(glGetUniformLocation(cull_prog_, "u_proj_00"), hiz_proj_00_);
        glUniform1f(glGetUniformLocation(cull_prog_, "u_proj_11"), hiz_proj_11_);
        glUniform1i(glGetUniformLocation(cull_prog_, "u_hiz_max_level"), hiz_max_level_);
        glBindTextureUnit(4, hiz_texture_);
    } else {
        glUniform1i(glGetUniformLocation(cull_prog_, "u_hiz_max_level"), 0);
    }

    glUniform1i(glGetUniformLocation(cull_prog_, "u_group_count"), group_count_);
    glUniform1iv(glGetUniformLocation(cull_prog_, "u_group_lod_counts[0]"),
        group_count_, group_lod_counts_);

    if (lod_threshold_count_ > 0) {
        glUniform1fv(glGetUniformLocation(cull_prog_, "u_lod_thresholds[0]"),
            lod_threshold_count_, lod_thresholds_);
    }

    GLuint groups = (static_cast<GLuint>(active_instances_) + 63) / 64;
    glDispatchCompute(groups, 1, 1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    cull_timer_.end();

    // --- Pass 2: Write indirect commands ---
    pack_timer_.begin();
    glUseProgram(pack_prog_);

    glUniform1uiv(glGetUniformLocation(pack_prog_, "u_slot_index_count"),
        total_slots_, slot_index_count_);
    glUniform1uiv(glGetUniformLocation(pack_prog_, "u_slot_first_index"),
        total_slots_, slot_first_index_);
    GLint base_vertex_ints[kMaxSlots];
    for (int i = 0; i < total_slots_; ++i) base_vertex_ints[i] = slot_base_vertex_[i];
    glUniform1iv(glGetUniformLocation(pack_prog_, "u_slot_base_vertex"),
        total_slots_, base_vertex_ints);
    glUniform1i(glGetUniformLocation(pack_prog_, "u_total_slots"), total_slots_);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, indirect_ssbo_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, counter_ssbo_);

    glDispatchCompute(1, 1, 1);

    glMemoryBarrier(GL_COMMAND_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    pack_timer_.end();

    // --- Draw visible instances ---
    draw_timer_.begin();
    glUseProgram(render_program);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indirect_ssbo_);

    if (has_indirect_params_) {
        glUniform1i(glGetUniformLocation(render_program, "u_use_multi_draw"), 1);
        glUniform1ui(glGetUniformLocation(render_program, "u_max_instances"),
            static_cast<GLuint>(max_instances_));
        GLuint draw_lod[kMaxSlots];
        for (int i = 0; i < total_slots_; ++i) draw_lod[i] = slot_lod_level_[i];
        glUniform1uiv(glGetUniformLocation(render_program, "u_draw_lod_level"),
            total_slots_, draw_lod);
        glUniform1i(glGetUniformLocation(render_program, "u_total_slots"), total_slots_);

        glBindVertexArray(merged_vao_);
        glBindBuffer(GL_PARAMETER_BUFFER, draw_count_ssbo_);
        glMultiDrawElementsIndirectCount(
            GL_TRIANGLES, GL_UNSIGNED_INT,
            nullptr, 0,
            static_cast<GLsizei>(total_slots_), 5 * sizeof(GLuint));
        glBindBuffer(GL_PARAMETER_BUFFER, 0);
    } else {
        glUniform1i(glGetUniformLocation(render_program, "u_use_multi_draw"), 0);
        for (int slot = 0; slot < total_slots_; ++slot) {
            GLuint offset = static_cast<GLuint>(slot * max_instances_);
            glUniform1ui(glGetUniformLocation(render_program, "u_lod_offset"), offset);
            glUniform1i(glGetUniformLocation(render_program, "u_lod_level"),
                static_cast<GLint>(slot_lod_level_[slot]));
            glBindVertexArray(merged_vao_);
            glDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT,
                reinterpret_cast<const void*>(static_cast<uintptr_t>(slot * 5 * sizeof(GLuint))));
        }
    }

    glBindVertexArray(0);
    draw_timer_.end();
    timer_started_ = true;

    if (readback_active_) {
        debug_readback();
    }
}

// --- Debug ---

void GpuPipeline::debug_readback() {
    GLuint raw[5 * kMaxSlots];
    glGetNamedBufferSubData(indirect_ssbo_, 0, sizeof(raw), raw);
    last_visible_count_ = 0;
    for (int i = 0; i < total_slots_; ++i) {
        last_lod_visible_[i] = raw[i * 5 + 1];
        last_visible_count_ += last_lod_visible_[i];
    }
}

} // namespace gfx
