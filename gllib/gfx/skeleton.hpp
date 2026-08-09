#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <string>
#include <vector>

namespace gfx {

struct Joint {
    std::string name;
    int parent = -1;
    glm::mat4 inverse_bind_matrix{1.0f};
};

class Skeleton {
public:
    Skeleton() = default;
    ~Skeleton();

    Skeleton(const Skeleton&) = delete;
    Skeleton& operator=(const Skeleton&) = delete;

    Skeleton(Skeleton&&) noexcept;
    Skeleton& operator=(Skeleton&&) noexcept;

    int joint_count() const { return static_cast<int>(joints_.size()); }
    Joint& joint(int i) { return joints_[i]; }
    const Joint& joint(int i) const { return joints_[i]; }

    void build(int num_joints,
               const std::vector<glm::mat4>& inverse_bind_matrices,
               const std::vector<int>& parent_indices,
               const std::vector<std::string>& joint_names);

    glm::mat4& joint_local_transform(int i) { return local_transforms_[i]; }
    const glm::mat4& joint_local_transform(int i) const { return local_transforms_[i]; }

    void set_default_pose(const std::vector<glm::vec3>& translations,
                          const std::vector<glm::quat>& rotations,
                          const std::vector<glm::vec3>& scales);

    glm::vec3 default_translation(int i) const { return default_pos_[i]; }
    glm::quat default_rotation(int i) const { return default_rot_[i]; }
    glm::vec3 default_scale(int i) const { return default_scl_[i]; }

    void reset_to_default_pose();

    void update();

    GLuint palette_ssbo() const { return palette_ssbo_; }
    const std::vector<glm::mat4>& bone_matrices() const { return bone_matrices_; }

private:
    std::vector<Joint> joints_;
    std::vector<glm::mat4> local_transforms_;
    std::vector<glm::mat4> bone_matrices_;
    GLuint palette_ssbo_ = 0;
    std::vector<glm::vec3> default_pos_;
    std::vector<glm::quat> default_rot_;
    std::vector<glm::vec3> default_scl_;
};

} // namespace gfx
