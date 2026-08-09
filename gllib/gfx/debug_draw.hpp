#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>

namespace gfx {

class DebugDraw {
public:
    DebugDraw();
    ~DebugDraw();

    DebugDraw(const DebugDraw&) = delete;
    DebugDraw& operator=(const DebugDraw&) = delete;

    DebugDraw(DebugDraw&& other) noexcept;
    DebugDraw& operator=(DebugDraw&& other) noexcept;

    void draw_line(const glm::vec3& from, const glm::vec3& to,
                   const glm::vec4& color);

    void draw_box(const glm::vec3& min, const glm::vec3& max,
                  const glm::vec4& color);

    void draw_sphere(const glm::vec3& center, float radius,
                     const glm::vec4& color, int segments = 16);

    void draw_frustum(const glm::mat4& inv_view_proj,
                      const glm::vec4& color);

    void draw_axis(const glm::mat4& transform, float size = 1.0f);

    void clear();

    void render(const glm::mat4& view_proj);

private:
    struct Vertex {
        glm::vec3 position;
        glm::vec4 color;
    };

    void push_line(const glm::vec3& a, const glm::vec3& b,
                   const glm::vec4& color);

    std::vector<Vertex> vertices_;

    struct Impl;
    Impl* impl_ = nullptr;
};

} // namespace gfx
