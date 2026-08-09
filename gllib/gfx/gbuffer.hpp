#pragma once

#include <glad/glad.h>

namespace gfx {

class GBuffer {
public:
    GBuffer() = default;
    ~GBuffer();

    GBuffer(const GBuffer&) = delete;
    GBuffer& operator=(const GBuffer&) = delete;

    GBuffer(GBuffer&&) noexcept;
    GBuffer& operator=(GBuffer&&) noexcept;

    void create(int width, int height);
    void bind_for_geometry();
    void bind_for_lighting(int albedo_unit, int normal_rm_unit,
                           int emissive_ao_unit, int depth_unit);

    int width() const { return width_; }
    int height() const { return height_; }

    GLuint albedo_handle() const { return albedo_; }
    GLuint normal_rm_handle() const { return normal_rm_; }
    GLuint emissive_ao_handle() const { return emissive_ao_; }
    GLuint depth_handle() const { return depth_; }
    GLuint fbo_handle() const { return fbo_; }

private:
    GLuint fbo_ = 0;
    GLuint albedo_ = 0;
    GLuint normal_rm_ = 0;
    GLuint emissive_ao_ = 0;
    GLuint depth_ = 0;
    int width_ = 0;
    int height_ = 0;
};

} // namespace gfx
