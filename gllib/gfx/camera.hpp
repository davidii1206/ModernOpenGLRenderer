#pragma once

#include <glm/glm.hpp>

namespace gfx {

class Camera {
public:
    Camera() = default;

    // --- Projection ---
    void perspective(float fov_y_degrees, float aspect, float near_clip, float far_clip);
    void ortho(float left, float right, float bottom, float top,
               float near_clip = -1.0f, float far_clip = 1.0f);
    void ortho_2d(float size, float aspect,
                  float near_clip = -1.0f, float far_clip = 1.0f);

    // --- View ---
    void look_at(const glm::vec3& eye, const glm::vec3& target,
                 const glm::vec3& up = glm::vec3(0, 1, 0));
    void set_position(const glm::vec3& pos);
    void set_target(const glm::vec3& target);
    void set_aspect(float aspect);

    // --- Controls ---
    void orbit(float delta_yaw, float delta_pitch);
    void pan(float delta_x, float delta_y);
    void zoom(float delta);

    void set_projection(const glm::mat4& proj) { projection_ = proj; }

    // --- Query ---
    const glm::mat4& view() const { return view_; }
    const glm::mat4& projection() const { return projection_; }
    glm::mat4 view_projection() const { return projection_ * view_; }

    const glm::vec3& position() const { return position_; }
    const glm::vec3& target() const { return target_; }

    float fov() const { return fov_; }
    float aspect() const { return aspect_; }
    float near_clip() const { return near_clip_; }
    float far_clip() const { return far_clip_; }

private:
    void recalc_view();
    void recalc_projection();

    // Projection
    bool is_perspective_ = true;
    float fov_ = 45.0f;
    float aspect_ = 16.0f / 9.0f;
    float near_clip_ = 0.1f;
    float far_clip_ = 100.0f;
    float ortho_left_ = -5.0f;
    float ortho_right_ = 5.0f;
    float ortho_bottom_ = -5.0f;
    float ortho_top_ = 5.0f;

    // View
    glm::vec3 position_ = glm::vec3(0, 0, 3);
    glm::vec3 target_ = glm::vec3(0, 0, 0);
    glm::vec3 up_ = glm::vec3(0, 1, 0);
    float yaw_ = 0.0f;
    float pitch_ = 0.0f;
    float distance_ = 3.0f;

    glm::mat4 view_ = glm::mat4(1);
    glm::mat4 projection_ = glm::mat4(1);
};

} // namespace gfx
