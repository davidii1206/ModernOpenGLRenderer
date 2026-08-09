#include "renderer.hpp"
#include "mesh.hpp"
#include "material.hpp"

#include <gl/state.hpp>

#include <algorithm>

namespace gfx {

void Renderer::set_clear_color(const float color[4]) {
    clear_color_[0] = color[0];
    clear_color_[1] = color[1];
    clear_color_[2] = color[2];
    clear_color_[3] = color[3];
}

void Renderer::set_clear_color(float r, float g, float b, float a) {
    clear_color_[0] = r;
    clear_color_[1] = g;
    clear_color_[2] = b;
    clear_color_[3] = a;
}

void Renderer::clear(GLbitfield mask) const {
    gl::clear_color(clear_color_[0], clear_color_[1],
                    clear_color_[2], clear_color_[3]);
    gl::clear(mask);
}

void Renderer::viewport(int x, int y, int width, int height) const {
    gl::viewport(x, y, width, height);
}

void Renderer::draw(const Mesh& mesh, const Material& material) const {
    material.bind();
    mesh.draw();
}

void Renderer::draw_instanced(const Mesh& mesh, const Material& material,
                               GLsizei instance_count) const
{
    material.bind();
    mesh.draw_instanced(instance_count);
}

void Renderer::draw(const DrawCommand& cmd) const {
    if (cmd.material) cmd.material->bind();
    if (cmd.mesh) {
        if (cmd.instance_count > 1) {
            cmd.mesh->draw_instanced(cmd.instance_count);
        } else {
            cmd.mesh->draw();
        }
    }
}

void Renderer::draw_batch(const std::vector<DrawCommand>& cmds) const {
    if (cmds.empty()) return;

    // Sort by material pointer to minimize state changes
    auto sorted = cmds;
    std::sort(sorted.begin(), sorted.end(),
              [](const DrawCommand& a, const DrawCommand& b) {
                  return a.material < b.material;
              });

    const Material* current_material = nullptr;
    for (const auto& cmd : sorted) {
        if (cmd.material != current_material) {
            if (cmd.material) cmd.material->bind();
            current_material = cmd.material;
        }
        if (cmd.mesh) {
            if (cmd.instance_count > 1) {
                cmd.mesh->draw_instanced(cmd.instance_count);
            } else {
                cmd.mesh->draw();
            }
        }
    }
}

} // namespace gfx
