#pragma once

#include <glad/glad.h>
#include <cstdint>
#include <vector>
#include <cstddef>

namespace gfx {

struct Vertex {
    float position[3];
    float normal[3];
    float texcoord[2];
    float tangent[4];
};

struct BoneWeight {
    uint16_t joints[4] = {};
    float weights[4] = {};
};

class Mesh {
public:
    Mesh() = default;
    ~Mesh();

    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

    Mesh(Mesh&& other) noexcept;
    Mesh& operator=(Mesh&& other) noexcept;

    void set_vertices(const std::vector<Vertex>& vertices);
    void set_indices(const std::vector<unsigned int>& indices);
    void set_bone_data(const std::vector<BoneWeight>& bone_data);
    void build();

    void draw() const;
    void draw_instanced(GLsizei instance_count) const;

    size_t vertex_count() const { return vertex_count_; }
    size_t index_count() const { return index_count_; }
    GLuint vao_handle() const { return vao_; }
    GLuint vbo_handle() const { return vbo_; }
    GLuint ebo_handle() const { return ebo_; }
    bool has_bone_data() const { return bone_vbo_ != 0; }
    GLuint bone_vbo_handle() const { return bone_vbo_; }

    const std::vector<Vertex>& vertices() const { return vertices_; }
    const std::vector<unsigned int>& indices() const { return indices_; }

private:
    GLuint vao_ = 0;
    GLuint vbo_ = 0;
    GLuint ebo_ = 0;
    GLuint bone_vbo_ = 0;
    size_t vertex_count_ = 0;
    size_t index_count_ = 0;
    std::vector<Vertex> vertices_;
    std::vector<unsigned int> indices_;
    std::vector<BoneWeight> bone_weights_;
};

} // namespace gfx
