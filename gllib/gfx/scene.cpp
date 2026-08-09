#include "scene.hpp"

#include "camera.hpp"
#include "mesh.hpp"
#include "material.hpp"
#include "renderer.hpp"
#include "skeleton.hpp"

#include <gl/state.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <stack>

namespace gfx {

// --- Node ---

Node::Node() = default;

Node::~Node() = default;

Node::Node(Node&& other) noexcept
    : local_transform_(other.local_transform_)
    , cached_world_(other.cached_world_)
    , dirty_(other.dirty_)
    , parent_(nullptr)
    , children_(std::move(other.children_))
    , mesh_(other.mesh_)
    , material_(other.material_)
    , name_(std::move(other.name_))
{
    for (auto& child : children_) {
        child->parent_ = this;
    }
    other.parent_ = nullptr;
    other.mesh_ = nullptr;
    other.material_ = nullptr;
    other.dirty_ = true;
}

Node& Node::operator=(Node&& other) noexcept {
    if (this != &other) {
        local_transform_ = other.local_transform_;
        cached_world_ = other.cached_world_;
        dirty_ = other.dirty_;
        mesh_ = other.mesh_;
        material_ = other.material_;
        name_ = std::move(other.name_);
        children_ = std::move(other.children_);
        for (auto& child : children_) {
            child->parent_ = this;
        }
        parent_ = nullptr;
        other.parent_ = nullptr;
        other.mesh_ = nullptr;
        other.material_ = nullptr;
        other.dirty_ = true;
    }
    return *this;
}

void Node::set_local_transform(const glm::mat4& m) {
    local_transform_ = m;
    mark_dirty();
}

glm::mat4 Node::world_transform() const {
    if (dirty_) {
        if (parent_) {
            cached_world_ = parent_->world_transform() * local_transform_;
        } else {
            cached_world_ = local_transform_;
        }
        dirty_ = false;
    }
    return cached_world_;
}

void Node::world_transform_dirty(bool dirty) const {
    dirty_ = dirty;
}

void Node::set_parent(Node* parent) {
    parent_ = parent;
    mark_dirty();
}

Node* Node::child(size_t index) const {
    return (index < children_.size()) ? children_[index].get() : nullptr;
}

void Node::add_child(std::unique_ptr<Node> child) {
    child->parent_ = this;
    child->mark_dirty();
    children_.push_back(std::move(child));
}

Node* Node::create_child(std::string_view name) {
    auto node = std::make_unique<Node>();
    auto* ptr = node.get();
    if (!name.empty()) ptr->set_name(name);
    add_child(std::move(node));
    return ptr;
}

std::unique_ptr<Node> Node::detach_child(size_t index) {
    if (index >= children_.size()) return nullptr;
    auto child = std::move(children_[index]);
    child->parent_ = nullptr;
    child->mark_dirty();
    children_.erase(children_.begin() + static_cast<ptrdiff_t>(index));
    return child;
}

void Node::mark_dirty() {
    dirty_ = true;
    for (auto& c : children_) {
        c->mark_dirty();
    }
}

// --- Scene ---

Scene::Scene()
    : root_(std::make_unique<Node>())
{
}

Node* Scene::create_node(std::string_view name) {
    auto node = std::make_unique<Node>();
    auto* ptr = node.get();
    if (!name.empty()) ptr->set_name(name);
    root_->add_child(std::move(node));
    return ptr;
}

void Scene::destroy_node(Node* node) {
    if (!node) return;

    // Find the node in the tree via DFS
    std::stack<Node*> stack;
    stack.push(root_.get());

    while (!stack.empty()) {
        auto* cur = stack.top();
        stack.pop();

        for (size_t i = 0; i < cur->child_count(); ++i) {
            if (cur->child(i) == node) {
                cur->detach_child(i);
                return;
            }
            stack.push(cur->child(i));
        }
    }
}

void Scene::update_transforms() {
    std::stack<Node*> stack;
    stack.push(root_.get());

    while (!stack.empty()) {
        auto* cur = stack.top();
        stack.pop();
        cur->world_transform();
        for (size_t i = 0; i < cur->child_count(); ++i) {
            stack.push(cur->child(i));
        }
    }
}

void Scene::draw(Renderer& renderer, const Camera& camera) {
    auto vp = camera.view_projection();
    std::stack<Node*> stack;
    stack.push(root_.get());

    while (!stack.empty()) {
        auto* cur = stack.top();
        stack.pop();

        if (cur->mesh() && cur->material()) {
            cur->material()->set_uniform("u_view_proj", vp);
            cur->material()->set_uniform("u_model", cur->world_transform());

            // Bind skeleton SSBO for skinned meshes
            if (cur->mesh()->has_bone_data() && cur->skeleton()) {
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4,
                                 cur->skeleton()->palette_ssbo());
                cur->material()->set_uniform("u_has_skin", 1);
            } else {
                cur->material()->set_uniform("u_has_skin", 0);
            }

            renderer.draw(*cur->mesh(), *cur->material());
        }

        for (size_t i = 0; i < cur->child_count(); ++i) {
            stack.push(cur->child(i));
        }
    }
}

} // namespace gfx
