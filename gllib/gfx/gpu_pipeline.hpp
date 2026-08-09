#pragma once

#include <glad/glad.h>
#include <gl/query.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include <vector>
#include <cstddef>

namespace gfx {

class Mesh;
class Camera;
class Model;

class GpuPipeline {
public:
    static constexpr int kMaxLods   = 8;
    static constexpr int kMaxGroups = 8;
    static constexpr int kMaxSlots  = kMaxGroups * kMaxLods; // 64

    struct InstanceData {
        glm::mat4 model;
        glm::vec4 bounding_sphere;
        GLuint    model_id  = 0;
        GLfloat   _pad[3]  = {};
    };

    explicit GpuPipeline(int max_instances = 65536);
    ~GpuPipeline();

    GpuPipeline(const GpuPipeline&) = delete;
    GpuPipeline& operator=(const GpuPipeline&) = delete;

    // Registration — call before first flush
    int  add_model(const Model& model, int lod_group_index = 0);
    void finalize();

    // Per-frame submission
    void draw(int model_handle, const glm::mat4& transform);
    void clear();
    void flush(const Camera& camera, GLuint render_program);

    // Hi-Z
    void set_hiz(GLuint texture, float proj_00, float proj_11,
                 int screen_w, int screen_h, int max_level);

    // LOD thresholds (max pixel size for each coarser LOD, up to kMaxLods-1 values)
    void set_lod_thresholds(const float* thresholds, int count);

    // Debug readback
    void debug_readback();
    int  last_visible_count() const { return last_visible_count_; }
    GLuint last_lod_visible(int slot) const { return last_lod_visible_[slot]; }
    void set_readback_active(bool active) { readback_active_ = active; }
    bool readback_active() const { return readback_active_; }
    int  slot_count() const { return total_slots_; }
    GLuint slot_index_count(int slot) const { return slot_index_count_[slot]; }
    int  max_instances() const { return max_instances_; }

    // GPU timers (microseconds)
    GLuint64 last_cull_time_us() const { return last_cull_time_ns_ / 1000; }
    GLuint64 last_pack_time_us() const { return last_pack_time_ns_ / 1000; }
    GLuint64 last_draw_time_us() const { return last_draw_time_ns_ / 1000; }
    GLuint64 last_total_time_us() const { return last_total_time_ns_ / 1000; }

private:
    struct ModelEntry {
        const Model* model = nullptr;
        int  lod_group_index = 0;
        int  group_id = -1;
        glm::vec4 local_bs = {};
    };

    void init_ssbo();
    void init_shaders();
    void init_indirect_params();
    void build_merged_vao();
    void cull_and_draw(const Camera& camera, GLuint render_program);
    void upload();
    int  slot_for(int group, int lod) const { return group * kMaxLods + lod; }

    bool finalized_ = false;
    int max_instances_;
    int group_count_ = 0;
    int total_slots_ = 0;
    int max_lod_count_ = 1;
    int active_instances_ = 0;

    std::vector<ModelEntry> registered_models_;
    std::vector<InstanceData> pending_draws_;
    std::vector<InstanceData> instances_;

    // GPU SSBOs
    GLuint instance_ssbo_  = 0;
    GLuint indirect_ssbo_  = 0;
    GLuint visible_ssbo_   = 0;
    GLuint counter_ssbo_   = 0;
    GLuint draw_count_ssbo_ = 0;
    GLuint cull_prog_      = 0;
    GLuint pack_prog_      = 0;
    bool   has_indirect_params_ = false;

    // Per-slot mesh info
    GLuint slot_index_count_[kMaxSlots]  = {};
    GLuint slot_first_index_[kMaxSlots]  = {};
    GLint  slot_base_vertex_[kMaxSlots]  = {};
    GLuint slot_lod_level_[kMaxSlots]    = {};
    int    group_lod_counts_[kMaxGroups] = {};

    // Merged VAO
    GLuint merged_vao_ = 0;
    GLuint merged_vbo_ = 0;
    GLuint merged_ebo_ = 0;

    glm::mat4 view_proj_;

    // Hi-Z
    GLuint hiz_texture_ = 0;
    float  hiz_proj_00_ = 0.0f;
    float  hiz_proj_11_ = 0.0f;
    int    hiz_screen_w_ = 0;
    int    hiz_screen_h_ = 0;
    int    hiz_max_level_ = 0;
    bool   hiz_active_ = false;

    // LOD thresholds
    float  lod_thresholds_[kMaxLods - 1] = {};
    int    lod_threshold_count_ = 0;

    // GPU timers
    gl::Query cull_timer_ = gl::Query(gl::QueryType::time_elapsed);
    gl::Query pack_timer_ = gl::Query(gl::QueryType::time_elapsed);
    gl::Query draw_timer_ = gl::Query(gl::QueryType::time_elapsed);
    bool      timer_started_ = false;
    GLuint64  last_cull_time_ns_ = 0;
    GLuint64  last_pack_time_ns_ = 0;
    GLuint64  last_draw_time_ns_ = 0;
    GLuint64  last_total_time_ns_ = 0;

    // Debug
    GLuint last_visible_count_ = 0;
    GLuint last_lod_visible_[kMaxSlots] = {};
    bool   readback_active_ = false;
};

} // namespace gfx
