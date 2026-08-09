#pragma once

#include <glad/glad.h>

namespace gfx {

class DepthPyramid {
public:
    DepthPyramid(int width, int height);
    ~DepthPyramid();

    DepthPyramid(const DepthPyramid&) = delete;
    DepthPyramid& operator=(const DepthPyramid&) = delete;

    DepthPyramid(DepthPyramid&& other) noexcept;
    DepthPyramid& operator=(DepthPyramid&& other) noexcept;

    void build(GLuint depth_texture);

    void bind(int unit) const;
    GLuint hiz_texture() const { return hiz_tex_; }
    int mip_levels() const { return mip_levels_; }
    int width() const { return width_; }
    int height() const { return height_; }

private:
    void init_shaders();

    int width_, height_, mip_levels_;
    GLuint hiz_tex_ = 0;
    GLuint copy_prog_ = 0;
    GLuint reduce_prog_ = 0;
};

} // namespace gfx
