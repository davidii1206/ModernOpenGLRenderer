#include "mesh.hpp"

#include <glad/glad.h>
#include <cstring>

namespace gfx {

Mesh::~Mesh() {
    if (vao_) glDeleteVertexArrays(1, &vao_);
    if (vbo_) glDeleteBuffers(1, &vbo_);
    if (ebo_) glDeleteBuffers(1, &ebo_);
    if (bone_vbo_) glDeleteBuffers(1, &bone_vbo_);
}

Mesh::Mesh(Mesh&& other) noexcept
    : vao_(other.vao_), vbo_(other.vbo_), ebo_(other.ebo_), bone_vbo_(other.bone_vbo_)
    , vertex_count_(other.vertex_count_), index_count_(other.index_count_)
    , vertices_(std::move(other.vertices_)), indices_(std::move(other.indices_))
    , bone_weights_(std::move(other.bone_weights_))
{
    other.vao_ = 0;
    other.vbo_ = 0;
    other.ebo_ = 0;
    other.bone_vbo_ = 0;
    other.vertex_count_ = 0;
    other.index_count_ = 0;
}

Mesh& Mesh::operator=(Mesh&& other) noexcept {
    if (this != &other) {
        if (vao_) glDeleteVertexArrays(1, &vao_);
        if (vbo_) glDeleteBuffers(1, &vbo_);
        if (ebo_) glDeleteBuffers(1, &ebo_);
        if (bone_vbo_) glDeleteBuffers(1, &bone_vbo_);
        vao_ = other.vao_;
        vbo_ = other.vbo_;
        ebo_ = other.ebo_;
        bone_vbo_ = other.bone_vbo_;
        vertex_count_ = other.vertex_count_;
        index_count_ = other.index_count_;
        vertices_ = std::move(other.vertices_);
        indices_ = std::move(other.indices_);
        bone_weights_ = std::move(other.bone_weights_);
        other.vao_ = 0;
        other.vbo_ = 0;
        other.ebo_ = 0;
        other.bone_vbo_ = 0;
        other.vertex_count_ = 0;
        other.index_count_ = 0;
    }
    return *this;
}

void Mesh::set_vertices(const std::vector<Vertex>& vertices) {
    vertices_ = vertices;
}

void Mesh::set_indices(const std::vector<unsigned int>& indices) {
    indices_ = indices;
}

void Mesh::set_bone_data(const std::vector<BoneWeight>& bone_data) {
    bone_weights_ = bone_data;
}

void Mesh::build() {
    if (vertices_.empty()) return;

    glCreateVertexArrays(1, &vao_);
    glCreateBuffers(1, &vbo_);

    auto vert_size = vertices_.size() * sizeof(Vertex);
    glNamedBufferData(vbo_, vert_size, vertices_.data(), GL_STATIC_DRAW);

    glVertexArrayVertexBuffer(vao_, 0, vbo_, 0, sizeof(Vertex));

    // position (location 0)
    glVertexArrayAttribFormat(vao_, 0, 3, GL_FLOAT, GL_FALSE, offsetof(Vertex, position));
    glVertexArrayAttribBinding(vao_, 0, 0);
    glEnableVertexArrayAttrib(vao_, 0);

    // normal (location 1)
    glVertexArrayAttribFormat(vao_, 1, 3, GL_FLOAT, GL_FALSE, offsetof(Vertex, normal));
    glVertexArrayAttribBinding(vao_, 1, 0);
    glEnableVertexArrayAttrib(vao_, 1);

    // texcoord (location 2)
    glVertexArrayAttribFormat(vao_, 2, 2, GL_FLOAT, GL_FALSE, offsetof(Vertex, texcoord));
    glVertexArrayAttribBinding(vao_, 2, 0);
    glEnableVertexArrayAttrib(vao_, 2);

    // tangent (location 3)
    glVertexArrayAttribFormat(vao_, 3, 4, GL_FLOAT, GL_FALSE, offsetof(Vertex, tangent));
    glVertexArrayAttribBinding(vao_, 3, 0);
    glEnableVertexArrayAttrib(vao_, 3);

    // Bone data (locations 4, 5) — separate VBO at binding index 1
    if (!bone_weights_.empty() && bone_weights_.size() == vertices_.size()) {
        glCreateBuffers(1, &bone_vbo_);
        auto bone_size = bone_weights_.size() * sizeof(BoneWeight);
        glNamedBufferData(bone_vbo_, bone_size, bone_weights_.data(), GL_STATIC_DRAW);

        glVertexArrayVertexBuffer(vao_, 1, bone_vbo_, 0, sizeof(BoneWeight));

        // joint indices (location 4) — 4 x unsigned short, integer attribute
        glVertexArrayAttribIFormat(vao_, 4, 4, GL_UNSIGNED_SHORT, offsetof(BoneWeight, joints));
        glVertexArrayAttribBinding(vao_, 4, 1);
        glEnableVertexArrayAttrib(vao_, 4);

        // joint weights (location 5) — 4 x float
        glVertexArrayAttribFormat(vao_, 5, 4, GL_FLOAT, GL_FALSE, offsetof(BoneWeight, weights));
        glVertexArrayAttribBinding(vao_, 5, 1);
        glEnableVertexArrayAttrib(vao_, 5);
    }

    if (!indices_.empty()) {
        glCreateBuffers(1, &ebo_);
        auto idx_size = indices_.size() * sizeof(unsigned int);
        glNamedBufferData(ebo_, idx_size, indices_.data(), GL_STATIC_DRAW);
        glVertexArrayElementBuffer(vao_, ebo_);
    }

    vertex_count_ = static_cast<GLsizei>(vertices_.size());
    index_count_ = static_cast<GLsizei>(indices_.size());
}

void Mesh::draw() const {
    glBindVertexArray(vao_);
    if (index_count_ > 0) {
        glDrawElements(GL_TRIANGLES, index_count_, GL_UNSIGNED_INT, nullptr);
    } else {
        glDrawArrays(GL_TRIANGLES, 0, vertex_count_);
    }
}

void Mesh::draw_instanced(GLsizei instance_count) const {
    glBindVertexArray(vao_);
    if (index_count_ > 0) {
        glDrawElementsInstanced(GL_TRIANGLES, index_count_, GL_UNSIGNED_INT, nullptr, instance_count);
    } else {
        glDrawArraysInstanced(GL_TRIANGLES, 0, vertex_count_, instance_count);
    }
}

} // namespace gfx
