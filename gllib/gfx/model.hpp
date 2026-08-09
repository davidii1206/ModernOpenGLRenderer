#pragma once

#include <glm/vec4.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace gfx {

class Mesh;
class Texture;
class Skeleton;
class AnimationClip;

enum AlphaMode : int {
    AlphaMode_Opaque = 0,
    AlphaMode_Mask  = 1,
    AlphaMode_Blend = 2,
};

struct ModelMaterialInfo {
    std::string name;
    int base_color_tex = -1;
    float base_color_factor[4] = {1, 1, 1, 1};
    int metallic_roughness_tex = -1;
    float metallic_factor = 1.0f;
    float roughness_factor = 1.0f;
    int normal_tex = -1;
    int occlusion_tex = -1;
    int emissive_tex = -1;
    float emissive_factor[3] = {0, 0, 0};
    AlphaMode alpha_mode = AlphaMode_Opaque;
    float alpha_cutoff = 0.5f;
    bool double_sided = false;   // glTF material.doubleSided (no backface culling)
};

struct LodGroup {
    std::string name;              // base name (without LOD suffix)
    std::vector<int> mesh_indices; // lod_level → meshes_[index]
};

class Model {
public:
    Model();
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    Model(Model&&) noexcept;
    Model& operator=(Model&&) noexcept;

    bool load(std::string_view path);

    size_t mesh_count() const { return meshes_.size(); }
    const Mesh& mesh(size_t i) const { return *meshes_[i]; }

    int mesh_material(size_t i) const { return mesh_material_map_[i]; }
    const std::string& mesh_name(size_t i) const { return mesh_names_[i]; }
    const glm::vec4& mesh_bounding_sphere(size_t i) const { return mesh_bounding_spheres_[i]; }

    size_t material_count() const { return materials_.size(); }
    const ModelMaterialInfo& material_info(size_t i) const { return materials_[i]; }

    size_t texture_count() const { return textures_.size(); }
    const std::shared_ptr<Texture>& texture(size_t i) const { return textures_[i]; }

    size_t lod_group_count() const { return lod_groups_.size(); }
    const LodGroup& lod_group(size_t i) const { return lod_groups_[i]; }

    bool has_skin() const { return skeleton_ != nullptr; }
    const Skeleton& skeleton() const { return *skeleton_; }
    Skeleton& skeleton() { return *skeleton_; }

    size_t animation_count() const { return animations_.size(); }
    const AnimationClip& animation(size_t i) const { return *animations_[i]; }
    AnimationClip& animation(size_t i) { return *animations_[i]; }

    void clear();

private:
    bool load_gltf(const std::string& path);
    void detect_lods();

    std::vector<std::unique_ptr<Mesh>> meshes_;
    std::vector<int> mesh_material_map_;
    std::vector<std::string> mesh_names_;
    std::vector<glm::vec4> mesh_bounding_spheres_;
    std::vector<ModelMaterialInfo> materials_;
    std::vector<std::shared_ptr<Texture>> textures_;
    std::vector<LodGroup> lod_groups_;
    std::unique_ptr<Skeleton> skeleton_;
    std::vector<std::unique_ptr<AnimationClip>> animations_;
};

} // namespace gfx
