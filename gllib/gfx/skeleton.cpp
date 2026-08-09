#include "skeleton.hpp"

#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <numeric>

namespace gfx {

Skeleton::~Skeleton() {
    if (palette_ssbo_) glDeleteBuffers(1, &palette_ssbo_);
}

Skeleton::Skeleton(Skeleton&& other) noexcept
    : joints_(std::move(other.joints_))
    , local_transforms_(std::move(other.local_transforms_))
    , bone_matrices_(std::move(other.bone_matrices_))
    , palette_ssbo_(other.palette_ssbo_)
{
    other.palette_ssbo_ = 0;
}

Skeleton& Skeleton::operator=(Skeleton&& other) noexcept {
    if (this != &other) {
        if (palette_ssbo_) glDeleteBuffers(1, &palette_ssbo_);
        joints_ = std::move(other.joints_);
        local_transforms_ = std::move(other.local_transforms_);
        bone_matrices_ = std::move(other.bone_matrices_);
        palette_ssbo_ = other.palette_ssbo_;
        other.palette_ssbo_ = 0;
    }
    return *this;
}

void Skeleton::build(int num_joints,
                     const std::vector<glm::mat4>& inverse_bind_matrices,
                     const std::vector<int>& parent_indices,
                     const std::vector<std::string>& joint_names)
{
    joints_.resize(num_joints);
    local_transforms_.assign(num_joints, glm::mat4(1.0f));
    bone_matrices_.resize(num_joints);
    default_pos_.assign(num_joints, glm::vec3(0.0f));
    default_rot_.assign(num_joints, glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
    default_scl_.assign(num_joints, glm::vec3(1.0f));

    for (int i = 0; i < num_joints; ++i) {
        joints_[i].name = joint_names[i];
        joints_[i].parent = parent_indices[i];
        joints_[i].inverse_bind_matrix = inverse_bind_matrices[i];
    }
}

void Skeleton::set_default_pose(const std::vector<glm::vec3>& translations,
                                const std::vector<glm::quat>& rotations,
                                const std::vector<glm::vec3>& scales)
{
    int n = joint_count();
    for (int i = 0; i < n && i < static_cast<int>(translations.size()); ++i)
        default_pos_[i] = translations[i];
    for (int i = 0; i < n && i < static_cast<int>(rotations.size()); ++i)
        default_rot_[i] = rotations[i];
    for (int i = 0; i < n && i < static_cast<int>(scales.size()); ++i)
        default_scl_[i] = scales[i];
}

void Skeleton::reset_to_default_pose() {
    int n = joint_count();
    local_transforms_.resize(n);
    for (int i = 0; i < n; ++i) {
        local_transforms_[i] =
            glm::translate(glm::mat4(1.0f), default_pos_[i]) *
            glm::mat4_cast(default_rot_[i]) *
            glm::scale(glm::mat4(1.0f), default_scl_[i]);
    }
}

void Skeleton::update() {
    int n = joint_count();
    if (n == 0) return;

    // Compute depths for topological ordering (walk to root)
    std::vector<int> depth(n, 0);
    for (int i = 0; i < n; ++i) {
        int d = 0, p = joints_[i].parent;
        while (p >= 0) { ++d; p = joints_[p].parent; }
        depth[i] = d;
    }

    // Sort joints by depth so parents are processed before children
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return depth[a] < depth[b]; });

    // Compute world-space transforms for each joint
    std::vector<glm::mat4> world(n);
    for (int idx : order) {
        world[idx] = local_transforms_[idx];
        int p = joints_[idx].parent;
        if (p >= 0) {
            world[idx] = world[p] * world[idx];
        }
    }

    // Compute bone matrices: world * inverse_bind
    for (int i = 0; i < n; ++i) {
        bone_matrices_[i] = world[i] * joints_[i].inverse_bind_matrix;
    }

    // Upload to SSBO
    if (!palette_ssbo_) {
        glCreateBuffers(1, &palette_ssbo_);
    }
    glNamedBufferData(palette_ssbo_,
                      static_cast<GLsizeiptr>(n) * sizeof(glm::mat4),
                      bone_matrices_.data(),
                      GL_DYNAMIC_DRAW);
}

} // namespace gfx
