#include "renderpass.hpp"

#include <gl/framebuffer.hpp>
#include <gl/state.hpp>

namespace gfx {

RenderPass::RenderPass()
    : desc_(Desc{})
{
}

RenderPass::RenderPass(const Desc& desc)
    : desc_(desc)
{
}

void RenderPass::begin() {
    if (desc_.framebuffer) {
        desc_.framebuffer->bind();
    } else {
        gl::Framebuffer::unbind(gl::FramebufferType::both);
    }

    gl::clear_color(desc_.clear_color.r, desc_.clear_color.g,
                    desc_.clear_color.b, desc_.clear_color.a);

    if (desc_.clear_depth != 1.0f) {
        glClearDepthf(desc_.clear_depth);
    }
    if (desc_.clear_stencil != 0) {
        glClearStencil(desc_.clear_stencil);
    }

    GLbitfield mask = 0;
    if (desc_.clear_color_attachment)   mask |= GL_COLOR_BUFFER_BIT;
    if (desc_.clear_depth_attachment)   mask |= GL_DEPTH_BUFFER_BIT;
    if (desc_.clear_stencil_attachment) mask |= GL_STENCIL_BUFFER_BIT;
    if (mask) gl::clear(mask);

    if (vp_[2] > 0 && vp_[3] > 0) {
        gl::viewport(vp_[0], vp_[1], vp_[2], vp_[3]);
    }

    began_ = true;
}

void RenderPass::end() {
    if (desc_.framebuffer) {
        gl::Framebuffer::unbind(gl::FramebufferType::both);
    }
    began_ = false;
}

void RenderPass::set_viewport(int x, int y, int w, int h) {
    vp_[0] = x;
    vp_[1] = y;
    vp_[2] = w;
    vp_[3] = h;
}

void RenderPass::set_clear_color(float r, float g, float b, float a) {
    desc_.clear_color = {r, g, b, a};
}

void RenderPass::set_clear_color(const glm::vec4& color) {
    desc_.clear_color = color;
}

} // namespace gfx
