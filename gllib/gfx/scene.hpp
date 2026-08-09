#pragma once

#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace gfx {

class Mesh;
class Material;
class Camera;
class Renderer;
class Skeleton;

class Node {
public:
    Node();
    ~Node();

    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    Node(Node&& other) noexcept;
    Node& operator=(Node&& other) noexcept;

    void set_local_transform(const glm::mat4& m);
    const glm::mat4& local_transform() const { return local_transform_; }

    glm::mat4 world_transform() const;
    void world_transform_dirty(bool dirty) const;

    void set_parent(Node* parent);
    Node* parent() const { return parent_; }

    size_t child_count() const { return children_.size(); }
    Node* child(size_t index) const;
    void add_child(std::unique_ptr<Node> child);
    Node* create_child(std::string_view name = {});
    std::unique_ptr<Node> detach_child(size_t index);

    void set_mesh(Mesh* mesh) { mesh_ = mesh; }
    Mesh* mesh() const { return mesh_; }

    void set_material(Material* material) { material_ = material; }
    Material* material() const { return material_; }

    void set_skeleton(Skeleton* skeleton) { skeleton_ = skeleton; }
    Skeleton* skeleton() const { return skeleton_; }

    void set_name(std::string_view name) { name_ = name; }
    const std::string& name() const { return name_; }

private:
    void mark_dirty();

    glm::mat4 local_transform_{1.0f};
    mutable glm::mat4 cached_world_{1.0f};
    mutable bool dirty_ = true;

    Node* parent_ = nullptr;
    std::vector<std::unique_ptr<Node>> children_;

    Mesh* mesh_ = nullptr;
    Material* material_ = nullptr;
    Skeleton* skeleton_ = nullptr;
    std::string name_;
};

class Scene {
public:
    Scene();
    ~Scene() = default;

    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;

    Scene(Scene&&) noexcept = default;
    Scene& operator=(Scene&&) noexcept = default;

    Node* root() { return root_.get(); }
    const Node* root() const { return root_.get(); }

    Node* create_node(std::string_view name = {});
    void destroy_node(Node* node);

    void update_transforms();

    void draw(Renderer& renderer, const Camera& camera);

private:
    std::unique_ptr<Node> root_;
};

} // namespace gfx
