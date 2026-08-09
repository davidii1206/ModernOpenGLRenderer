#include "camera.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>
#include <cmath>

namespace gfx {

// --- Projection ---

void Camera::perspective(float fov_y_degrees, float aspect,
                         float near_clip, float far_clip)
{
    is_perspective_ = true;
    fov_ = fov_y_degrees;
    aspect_ = aspect;
    near_clip_ = near_clip;
    far_clip_ = far_clip;
    recalc_projection();
}

void Camera::ortho(float left, float right, float bottom, float top,
                   float near_clip, float far_clip)
{
    is_perspective_ = false;
    ortho_left_ = left;
    ortho_right_ = right;
    ortho_bottom_ = bottom;
    ortho_top_ = top;
    near_clip_ = near_clip;
    far_clip_ = far_clip;
    recalc_projection();
}

void Camera::ortho_2d(float size, float aspect,
                      float near_clip, float far_clip)
{
    float half_h = size;
    float half_w = size * aspect;
    ortho(-half_w, half_w, -half_h, half_h, near_clip, far_clip);
}

// --- View ---

void Camera::look_at(const glm::vec3& eye, const glm::vec3& target,
                     const glm::vec3& up)
{
    position_ = eye;
    target_ = target;
    up_ = up;

    glm::vec3 diff = eye - target;
    distance_ = glm::length(diff);
    if (distance_ > 0.001f) {
        glm::vec3 dir = diff / distance_;
        yaw_ = std::atan2(dir.z, dir.x);
        pitch_ = std::asin(glm::clamp(dir.y, -1.0f, 1.0f));
    }

    recalc_view();
}

void Camera::set_position(const glm::vec3& pos) {
    position_ = pos;
    glm::vec3 diff = position_ - target_;
    distance_ = glm::length(diff);
    if (distance_ > 0.001f) {
        glm::vec3 dir = diff / distance_;
        yaw_ = std::atan2(dir.z, dir.x);
        pitch_ = std::asin(glm::clamp(dir.y, -1.0f, 1.0f));
    }
    recalc_view();
}

void Camera::set_target(const glm::vec3& target) {
    target_ = target;
    recalc_view();
}

void Camera::set_aspect(float aspect) {
    aspect_ = aspect;
    recalc_projection();
}

// --- Controls ---

void Camera::orbit(float delta_yaw, float delta_pitch) {
    yaw_ += delta_yaw;
    pitch_ += delta_pitch;
    pitch_ = glm::clamp(pitch_, -glm::half_pi<float>() + 0.01f,
                        glm::half_pi<float>() - 0.01f);

    glm::vec3 dir;
    dir.x = std::cos(yaw_) * std::cos(pitch_);
    dir.y = std::sin(pitch_);
    dir.z = std::sin(yaw_) * std::cos(pitch_);

    position_ = target_ + dir * distance_;
    recalc_view();
}

void Camera::pan(float delta_x, float delta_y) {
    glm::vec3 dir = glm::normalize(target_ - position_);
    glm::vec3 right = glm::normalize(glm::cross(dir, up_));
    glm::vec3 up = glm::normalize(glm::cross(right, dir));
    glm::vec3 offset = (right * delta_x + up * delta_y) * distance_ * 0.2f;
    position_ += offset;
    target_ += offset;
    recalc_view();
}

void Camera::zoom(float delta) {
    distance_ = std::max(0.1f, distance_ + delta);
    glm::vec3 dir = (position_ - target_) / glm::length(position_ - target_);
    position_ = target_ + dir * distance_;
    recalc_view();
}

// --- Internal ---

void Camera::recalc_view() {
    view_ = glm::lookAt(position_, target_, up_);
}

void Camera::recalc_projection() {
    if (is_perspective_) {
        projection_ = glm::perspective(glm::radians(fov_), aspect_,
                                        near_clip_, far_clip_);
    } else {
        projection_ = glm::ortho(ortho_left_, ortho_right_,
                                 ortho_bottom_, ortho_top_,
                                 near_clip_, far_clip_);
    }
}

} // namespace gfx
