#pragma once

#include <glm/glm.hpp>
#include <memory>

namespace gl {
class Framebuffer;
} // namespace gl

namespace gfx {

class RenderPass {
public:
    struct Desc {
        std::shared_ptr<gl::Framebuffer> framebuffer;
        glm::vec4 clear_color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        float clear_depth = 1.0f;
        int clear_stencil = 0;
        bool clear_color_attachment = true;
        bool clear_depth_attachment = true;
        bool clear_stencil_attachment = true;
    };

    RenderPass();
    explicit RenderPass(const Desc& desc);
    ~RenderPass() = default;

    RenderPass(const RenderPass&) = delete;
    RenderPass& operator=(const RenderPass&) = delete;

    RenderPass(RenderPass&&) noexcept = default;
    RenderPass& operator=(RenderPass&&) noexcept = default;

    void begin();
    void end();

    void set_viewport(int x, int y, int w, int h);
    void set_clear_color(float r, float g, float b, float a);
    void set_clear_color(const glm::vec4& color);

    int viewport_x() const { return vp_[0]; }
    int viewport_y() const { return vp_[1]; }
    int viewport_width() const { return vp_[2]; }
    int viewport_height() const { return vp_[3]; }

    const Desc& desc() const { return desc_; }

private:
    Desc desc_;
    int vp_[4] = {0, 0, 0, 0};
    bool began_ = false;
};

} // namespace gfx
