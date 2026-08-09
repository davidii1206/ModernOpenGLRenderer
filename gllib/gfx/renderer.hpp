#pragma once

#include <glad/glad.h>
#include <vector>

namespace gfx {

class Mesh;
class Material;

struct DrawCommand {
    const Mesh* mesh = nullptr;
    const Material* material = nullptr;
    GLsizei instance_count = 1;
};

class Renderer {
public:
    Renderer() = default;
    ~Renderer() = default;

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    Renderer(Renderer&& other) noexcept = default;
    Renderer& operator=(Renderer&& other) noexcept = default;

    void set_clear_color(const float color[4]);
    void set_clear_color(float r, float g, float b, float a);

    void clear(GLbitfield mask = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) const;
    void viewport(int x, int y, int width, int height) const;

    void draw(const Mesh& mesh, const Material& material) const;
    void draw_instanced(const Mesh& mesh, const Material& material,
                        GLsizei instance_count) const;
    void draw(const DrawCommand& cmd) const;
    void draw_batch(const std::vector<DrawCommand>& cmds) const;

private:
    float clear_color_[4] = {0.0f, 0.0f, 0.0f, 1.0f};
};

} // namespace gfx
