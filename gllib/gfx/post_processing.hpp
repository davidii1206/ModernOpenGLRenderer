#pragma once

#include <glad/glad.h>

namespace gl {
class Framebuffer;
class Texture;
class Renderbuffer;
class Program;
class VertexArray;
class Buffer;
} // namespace gl

namespace gfx {

class Camera;

class PostProcessing {
public:
    PostProcessing(int width, int height);
    ~PostProcessing();

    PostProcessing(const PostProcessing&) = delete;
    PostProcessing& operator=(const PostProcessing&) = delete;

    PostProcessing(PostProcessing&&) noexcept;
    PostProcessing& operator=(PostProcessing&&) noexcept;

    void begin(bool clear = true);
    void end();

    void resize(int width, int height);

    void set_exposure(float e);
    void set_bloom_enabled(bool enabled);
    void set_bloom_intensity(float intensity);
    void set_bloom_threshold(float threshold);

    GLuint hdr_color_handle() const;

private:
    void init_fbos();
    void init_shaders();
    void init_quad();
    void destroy_fbos();
    void mark_moved();
    void blur(GLuint input, bool horizontal);
    void bright_extract(GLuint input);

    int width_, height_;
    float exposure_ = 1.0f;
    float bloom_intensity_ = 0.5f;
    float bloom_threshold_ = 1.0f;
    bool bloom_enabled_ = true;

    gl::Framebuffer* hdr_fbo_ = nullptr;
    gl::Texture* hdr_color_ = nullptr;
    gl::Renderbuffer* hdr_depth_ = nullptr;

    gl::Framebuffer* bloom_fbo_[2] = {};
    gl::Texture* bloom_tex_[2] = {};

    gl::Program* bright_prog_ = nullptr;
    gl::Program* blur_prog_ = nullptr;
    gl::Program* composite_prog_ = nullptr;

    gl::VertexArray* quad_vao_ = nullptr;
    gl::Buffer* quad_vbo_ = nullptr;
    gl::Buffer* quad_ebo_ = nullptr;
};

} // namespace gfx
