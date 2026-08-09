#include "debug_draw.hpp"

#include <gl/buffer.hpp>
#include <gl/program.hpp>
#include <gl/shader.hpp>
#include <gl/state.hpp>
#include <gl/vertex_array.hpp>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/constants.hpp>

#include <cstring>
#include <vector>

namespace gfx {

// ── Embedded shaders ──

static const char* vert_src = R"(
#version 460 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec4 a_color;
uniform mat4 u_view_proj;
out vec4 v_color;
void main() {
    gl_Position = u_view_proj * vec4(a_pos, 1.0);
    v_color = a_color;
}
)";

static const char* frag_src = R"(
#version 460 core
in vec4 v_color;
out vec4 frag_color;
void main() {
    frag_color = v_color;
}
)";

// ── Pimpl ──

struct DebugDraw::Impl {
    gl::Buffer vbo{gl::BufferType::vertex, gl::BufferUsage::dynamic_draw};
    gl::VertexArray vao;
    gl::Program prog;
    bool programs_ok = false;

    Impl() {
        gl::Shader vs(gl::ShaderType::vertex, vert_src);
        gl::Shader fs(gl::ShaderType::fragment, frag_src);
        if (!vs.compiled() || !fs.compiled()) return;

        prog.attach(vs);
        prog.attach(fs);
        if (!prog.link()) return;
        programs_ok = true;

        vao.bind();
        // Vertex attribs will be set up at render time when vbo is bound
        gl::VertexArray::unbind();
    }
};

// ── Construction ──

DebugDraw::DebugDraw()
    : impl_(new Impl())
{}

DebugDraw::~DebugDraw() {
    delete impl_;
}

DebugDraw::DebugDraw(DebugDraw&& other) noexcept
    : vertices_(std::move(other.vertices_))
    , impl_(other.impl_)
{
    other.impl_ = nullptr;
}

DebugDraw& DebugDraw::operator=(DebugDraw&& other) noexcept {
    if (this != &other) {
        vertices_ = std::move(other.vertices_);
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

// ── Internal ──

void DebugDraw::push_line(const glm::vec3& a, const glm::vec3& b,
                           const glm::vec4& color) {
    vertices_.push_back({a, color});
    vertices_.push_back({b, color});
}

// ── Draw primitives ──

void DebugDraw::draw_line(const glm::vec3& from, const glm::vec3& to,
                           const glm::vec4& color) {
    push_line(from, to, color);
}

void DebugDraw::draw_box(const glm::vec3& min, const glm::vec3& max,
                          const glm::vec4& color) {
    // 8 corners
    glm::vec3 c[8] = {
        {min.x, min.y, min.z}, {max.x, min.y, min.z},
        {max.x, min.y, max.z}, {min.x, min.y, max.z},
        {min.x, max.y, min.z}, {max.x, max.y, min.z},
        {max.x, max.y, max.z}, {min.x, max.y, max.z},
    };
    // 12 edges (bottom face, top face, verticals)
    int edges[][2] = {
        {0,1}, {1,2}, {2,3}, {3,0},
        {4,5}, {5,6}, {6,7}, {7,4},
        {0,4}, {1,5}, {2,6}, {3,7},
    };
    for (auto& e : edges) {
        push_line(c[e[0]], c[e[1]], color);
    }
}

void DebugDraw::draw_sphere(const glm::vec3& center, float radius,
                             const glm::vec4& color, int segments) {
    // Latitude rings (horizontal circles at each step)
    for (int j = 1; j < segments / 2; ++j) {
        float theta = float(j) / (segments / 2) * glm::pi<float>();
        float r = radius * std::sin(theta);
        float y = radius * std::cos(theta);
        for (int i = 0; i < segments; ++i) {
            float a0 = float(i) / segments * glm::two_pi<float>();
            float a1 = float(i + 1) / segments * glm::two_pi<float>();
            glm::vec3 p0 = {r * std::cos(a0) + center.x,
                            y + center.y,
                            r * std::sin(a0) + center.z};
            glm::vec3 p1 = {r * std::cos(a1) + center.x,
                            y + center.y,
                            r * std::sin(a1) + center.z};
            push_line(p0, p1, color);
        }
    }

    // Longitude arcs (vertical half-circles around Y)
    for (int i = 0; i < segments / 2; ++i) {
        float angle = float(i) / (segments / 2) * glm::pi<float>();
        float ca = std::cos(angle), sa = std::sin(angle);
        for (int j = 0; j < segments; ++j) {
            float a0 = float(j) / segments * glm::two_pi<float>();
            float a1 = float(j + 1) / segments * glm::two_pi<float>();
            float x0 = radius * std::sin(a0);
            float y0 = radius * std::cos(a0);
            float x1 = radius * std::sin(a1);
            float y1 = radius * std::cos(a1);
            glm::vec3 p0 = {x0 * ca + center.x, y0 + center.y, x0 * sa + center.z};
            glm::vec3 p1 = {x1 * ca + center.x, y1 + center.y, x1 * sa + center.z};
            push_line(p0, p1, color);
        }
    }
}

void DebugDraw::draw_frustum(const glm::mat4& inv_view_proj,
                              const glm::vec4& color) {
    // 8 corners of NDC [-1, 1]^3, unprojected to world space
    glm::vec4 ndc_corners[8] = {
        {-1,-1,-1,1}, { 1,-1,-1,1}, { 1,-1, 1,1}, {-1,-1, 1,1},
        {-1, 1,-1,1}, { 1, 1,-1,1}, { 1, 1, 1,1}, {-1, 1, 1,1},
    };
    glm::vec3 world[8];
    for (int i = 0; i < 8; ++i) {
        glm::vec4 p = inv_view_proj * ndc_corners[i];
        world[i] = glm::vec3(p) / p.w;
    }

    int edges[][2] = {
        {0,1}, {1,2}, {2,3}, {3,0}, // near plane
        {4,5}, {5,6}, {6,7}, {7,4}, // far plane
        {0,4}, {1,5}, {2,6}, {3,7}, // connections
    };
    for (auto& e : edges) {
        push_line(world[e[0]], world[e[1]], color);
    }
}

void DebugDraw::draw_axis(const glm::mat4& transform, float size) {
    glm::vec3 origin = glm::vec3(transform[3]);
    glm::vec3 right(transform[0]);
    glm::vec3 up(transform[1]);
    glm::vec3 fwd(transform[2]);
    push_line(origin, origin + right * size, {1,0,0,1});
    push_line(origin, origin + up * size, {0,1,0,1});
    push_line(origin, origin + fwd * size, {0,0,1,1});
}

// ── Clear / Render ──

void DebugDraw::clear() {
    vertices_.clear();
}

void DebugDraw::render(const glm::mat4& view_proj) {
    if (!impl_ || !impl_->programs_ok || vertices_.empty()) {
        vertices_.clear();
        return;
    }

    auto& vbo = impl_->vbo;
    auto& vao = impl_->vao;
    auto& prog = impl_->prog;

    // Upload vertices
    size_t bytes = vertices_.size() * sizeof(Vertex);
    if (bytes > vbo.size()) {
        vbo.data(vertices_.data(), bytes);
    } else {
        vbo.sub_data(vertices_.data(), 0, bytes);
    }

    // Set up VAO
    vao.bind();
    vbo.bind();
    vao.attrib_pointer(0, 3, GL_FLOAT, false, sizeof(Vertex),
                       reinterpret_cast<const void*>(offsetof(Vertex, position)));
    vao.enable_attrib(0);
    vao.attrib_pointer(1, 4, GL_FLOAT, false, sizeof(Vertex),
                       reinterpret_cast<const void*>(offsetof(Vertex, color)));
    vao.enable_attrib(1);

    // Draw
    prog.use();
    GLint loc = prog.uniform_location("u_view_proj");
    if (loc >= 0) prog.uniform_matrix4fv(loc, &view_proj[0][0]);

    gl::draw_arrays(GL_LINES, 0, static_cast<GLsizei>(vertices_.size()));

    gl::VertexArray::unbind();
    vertices_.clear();
}

} // namespace gfx
