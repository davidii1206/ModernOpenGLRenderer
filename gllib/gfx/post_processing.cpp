#include "post_processing.hpp"

#include <gl/buffer.hpp>
#include <gl/framebuffer.hpp>
#include <gl/program.hpp>
#include <gl/renderbuffer.hpp>
#include <gl/shader.hpp>
#include <gl/state.hpp>
#include <gl/texture.hpp>
#include <gl/vertex_array.hpp>

namespace gfx {

static const char* quad_vert_src = R"(
#version 460 core
layout (location = 0) in vec2 a_pos;
layout (location = 1) in vec2 a_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_uv = a_uv;
}
)";

static const char* bright_frag_src = R"(
#version 460 core
uniform sampler2D u_color;
uniform float u_threshold;
in vec2 v_uv;
out vec4 frag_color;
void main() {
    vec3 c = texture(u_color, v_uv).rgb;
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    float brightness = l - u_threshold;
    frag_color = vec4(c * max(brightness / max(l, 0.001), 0.0), 1.0);
}
)";

static const char* blur_frag_src = R"(
#version 460 core
uniform sampler2D u_input;
uniform vec2 u_direction;
uniform vec2 u_texel_size;
in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec2 ts = u_texel_size * u_direction;
    vec4 col = texture(u_input, v_uv) * 0.2270270270;
    col += texture(u_input, v_uv + ts * 1.0) * 0.3162162162;
    col += texture(u_input, v_uv - ts * 1.0) * 0.3162162162;
    col += texture(u_input, v_uv + ts * 2.0) * 0.0702702703;
    col += texture(u_input, v_uv - ts * 2.0) * 0.0702702703;
    frag_color = col;
}
)";

static const char* composite_frag_src = R"(
#version 460 core
uniform sampler2D u_hdr;
uniform sampler2D u_bloom;
uniform float u_exposure;
uniform float u_bloom_intensity;
in vec2 v_uv;
out vec4 frag_color;

vec3 aces(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 hdr = texture(u_hdr, v_uv).rgb * u_exposure;
    vec3 bloom = texture(u_bloom, v_uv).rgb * u_bloom_intensity;
    vec3 col = aces(hdr + bloom);
    frag_color = vec4(col, 1.0);
}
)";

PostProcessing::PostProcessing(int width, int height)
    : width_(width)
    , height_(height)
{
    init_quad();
    init_fbos();
    init_shaders();
}

PostProcessing::~PostProcessing() {
    destroy_fbos();
    delete bright_prog_;
    delete blur_prog_;
    delete composite_prog_;
    delete quad_vao_;
    delete quad_vbo_;
    delete quad_ebo_;
}

PostProcessing::PostProcessing(PostProcessing&& other) noexcept
    : width_(other.width_), height_(other.height_)
    , exposure_(other.exposure_)
    , bloom_intensity_(other.bloom_intensity_)
    , bloom_threshold_(other.bloom_threshold_)
    , bloom_enabled_(other.bloom_enabled_)
    , hdr_fbo_(other.hdr_fbo_), hdr_color_(other.hdr_color_), hdr_depth_(other.hdr_depth_)
    , bright_prog_(other.bright_prog_), blur_prog_(other.blur_prog_), composite_prog_(other.composite_prog_)
    , quad_vao_(other.quad_vao_), quad_vbo_(other.quad_vbo_), quad_ebo_(other.quad_ebo_)
{
    bloom_fbo_[0] = other.bloom_fbo_[0];
    bloom_fbo_[1] = other.bloom_fbo_[1];
    bloom_tex_[0] = other.bloom_tex_[0];
    bloom_tex_[1] = other.bloom_tex_[1];
    other.mark_moved();
}

PostProcessing& PostProcessing::operator=(PostProcessing&& other) noexcept {
    if (this != &other) {
        destroy_fbos();
        delete bright_prog_; delete blur_prog_; delete composite_prog_;
        delete quad_vao_; delete quad_vbo_; delete quad_ebo_;

        width_ = other.width_; height_ = other.height_;
        exposure_ = other.exposure_;
        bloom_intensity_ = other.bloom_intensity_;
        bloom_threshold_ = other.bloom_threshold_;
        bloom_enabled_ = other.bloom_enabled_;
        hdr_fbo_ = other.hdr_fbo_; hdr_color_ = other.hdr_color_; hdr_depth_ = other.hdr_depth_;
        bright_prog_ = other.bright_prog_; blur_prog_ = other.blur_prog_; composite_prog_ = other.composite_prog_;
        quad_vao_ = other.quad_vao_; quad_vbo_ = other.quad_vbo_; quad_ebo_ = other.quad_ebo_;
        bloom_fbo_[0] = other.bloom_fbo_[0]; bloom_fbo_[1] = other.bloom_fbo_[1];
        bloom_tex_[0] = other.bloom_tex_[0]; bloom_tex_[1] = other.bloom_tex_[1];
        other.mark_moved();
    }
    return *this;
}

void PostProcessing::mark_moved() {
    hdr_fbo_ = nullptr; hdr_color_ = nullptr; hdr_depth_ = nullptr;
    bright_prog_ = blur_prog_ = composite_prog_ = nullptr;
    quad_vao_ = nullptr; quad_vbo_ = nullptr; quad_ebo_ = nullptr;
    bloom_fbo_[0] = bloom_fbo_[1] = nullptr;
    bloom_tex_[0] = bloom_tex_[1] = nullptr;
}

void PostProcessing::resize(int width, int height) {
    if (width == width_ && height == height_) return;
    width_ = width;
    height_ = height;
    destroy_fbos();
    init_fbos();
}

void PostProcessing::set_exposure(float e) { exposure_ = e; }
void PostProcessing::set_bloom_enabled(bool e) { bloom_enabled_ = e; }
void PostProcessing::set_bloom_intensity(float i) { bloom_intensity_ = i; }
void PostProcessing::set_bloom_threshold(float t) { bloom_threshold_ = t; }

GLuint PostProcessing::hdr_color_handle() const {
    return hdr_color_ ? hdr_color_->handle() : 0;
}

void PostProcessing::init_quad() {
    float verts[] = {
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,
    };
    unsigned int idx[] = {0, 1, 2, 0, 2, 3};

    quad_vbo_ = new gl::Buffer(gl::BufferType::vertex);
    quad_vbo_->data(verts, sizeof(verts));
    quad_ebo_ = new gl::Buffer(gl::BufferType::index);
    quad_ebo_->data(idx, sizeof(idx));

    quad_vao_ = new gl::VertexArray;
    quad_vao_->bind();
    quad_vbo_->bind();
    quad_ebo_->bind();
    quad_vao_->attrib_pointer(0, 2, GL_FLOAT, false, 16, (void*)0);
    quad_vao_->enable_attrib(0);
    quad_vao_->attrib_pointer(1, 2, GL_FLOAT, false, 16, (void*)8);
    quad_vao_->enable_attrib(1);
    gl::VertexArray::unbind();
}

void PostProcessing::init_fbos() {
    int w = width_, h = height_;

    // HDR framebuffer
    hdr_color_ = new gl::Texture(gl::TextureType::tex_2d);
    hdr_color_->bind();
    hdr_color_->image_2d(0, GL_RGBA16F, w, h, GL_RGBA, GL_FLOAT, nullptr);
    hdr_color_->parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    hdr_color_->parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    hdr_color_->parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    hdr_color_->parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    hdr_depth_ = new gl::Renderbuffer;
    hdr_depth_->bind();
    hdr_depth_->storage(GL_DEPTH_COMPONENT24, w, h);
    gl::Renderbuffer::unbind();

    hdr_fbo_ = new gl::Framebuffer;
    hdr_fbo_->bind();
    hdr_fbo_->attach_texture(GL_COLOR_ATTACHMENT0, *hdr_color_);
    hdr_fbo_->attach_renderbuffer(GL_DEPTH_ATTACHMENT, *hdr_depth_);
    hdr_fbo_->check();
    gl::Framebuffer::unbind(gl::FramebufferType::both);

    // Bloom framebuffers (half-res for performance)
    int bw = w / 2, bh = h / 2;
    if (bw < 1) bw = 1;
    if (bh < 1) bh = 1;

    for (int i = 0; i < 2; ++i) {
        bloom_tex_[i] = new gl::Texture(gl::TextureType::tex_2d);
        bloom_tex_[i]->bind();
        bloom_tex_[i]->image_2d(0, GL_RGBA16F, bw, bh, GL_RGBA, GL_FLOAT, nullptr);
        bloom_tex_[i]->parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        bloom_tex_[i]->parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        bloom_tex_[i]->parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        bloom_tex_[i]->parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        bloom_fbo_[i] = new gl::Framebuffer;
        bloom_fbo_[i]->bind();
        bloom_fbo_[i]->attach_texture(GL_COLOR_ATTACHMENT0, *bloom_tex_[i]);
        bloom_fbo_[i]->check();
        gl::Framebuffer::unbind(gl::FramebufferType::both);
    }
}

void PostProcessing::destroy_fbos() {
    delete hdr_fbo_;
    delete hdr_color_;
    delete hdr_depth_;
    for (int i = 0; i < 2; ++i) {
        delete bloom_fbo_[i];
        delete bloom_tex_[i];
    }
}

void PostProcessing::init_shaders() {
    auto make_prog = [](const char* vert, const char* frag) -> gl::Program* {
        gl::Shader vs(gl::ShaderType::vertex, vert);
        gl::Shader fs(gl::ShaderType::fragment, frag);
        if (!vs.compiled() || !fs.compiled()) return nullptr;
        auto* p = new gl::Program;
        p->attach(vs);
        p->attach(fs);
        if (!p->link()) { delete p; return nullptr; }
        return p;
    };
    bright_prog_ = make_prog(quad_vert_src, bright_frag_src);
    blur_prog_ = make_prog(quad_vert_src, blur_frag_src);
    composite_prog_ = make_prog(quad_vert_src, composite_frag_src);
}

void PostProcessing::begin(bool clear_scene) {
    if (!hdr_fbo_) return;
    hdr_fbo_->bind();
    if (clear_scene) {
        gl::clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
}

void PostProcessing::end() {
    if (!composite_prog_) return;

    if (bloom_enabled_ && bright_prog_ && blur_prog_) {
        bright_extract(hdr_color_->handle());
        blur(bloom_tex_[0]->handle(), true);
        blur(bloom_tex_[1]->handle(), false);
    }

    // Composite to default framebuffer
    gl::Framebuffer::unbind(gl::FramebufferType::both);
    gl::viewport(0, 0, width_, height_);
    gl::disable(GL_DEPTH_TEST);
    composite_prog_->use();
    glBindTextureUnit(0, hdr_color_->handle());
    glBindTextureUnit(1, bloom_enabled_ ? bloom_tex_[0]->handle() : 0);

    auto set = [&](const char* name, auto val, auto&& setter) {
        GLint loc = composite_prog_->uniform_location(name);
        if (loc >= 0) setter(loc, val);
    };
    set("u_hdr", 0, [&](GLint l, int v) { composite_prog_->uniform1i(l, v); });
    set("u_bloom", 1, [&](GLint l, int v) { composite_prog_->uniform1i(l, v); });
    set("u_exposure", exposure_, [&](GLint l, float v) { composite_prog_->uniform1f(l, v); });
    set("u_bloom_intensity", bloom_enabled_ ? bloom_intensity_ : 0.0f,
        [&](GLint l, float v) { composite_prog_->uniform1f(l, v); });

    quad_vao_->bind();
    gl::draw_elements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
}

void PostProcessing::bright_extract(GLuint input) {
    bright_prog_->use();
    glBindTextureUnit(0, input);

    GLint loc;
    loc = bright_prog_->uniform_location("u_color");
    if (loc >= 0) bright_prog_->uniform1i(loc, 0);
    loc = bright_prog_->uniform_location("u_threshold");
    if (loc >= 0) bright_prog_->uniform1f(loc, bloom_threshold_);

    bloom_fbo_[0]->bind();
    gl::viewport(0, 0, width_ / 2, height_ / 2);
    gl::clear(GL_COLOR_BUFFER_BIT);
    quad_vao_->bind();
    gl::draw_elements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
}

void PostProcessing::blur(GLuint input, bool horizontal) {
    blur_prog_->use();
    glBindTextureUnit(0, input);

    GLint loc;
    loc = blur_prog_->uniform_location("u_input");
    if (loc >= 0) blur_prog_->uniform1i(loc, 0);

    int bw = width_ / 2, bh = height_ / 2;
    if (bw < 1) bw = 1;
    if (bh < 1) bh = 1;

    loc = blur_prog_->uniform_location("u_texel_size");
    if (loc >= 0) blur_prog_->uniform2f(loc, 1.0f / bw, 1.0f / bh);

    loc = blur_prog_->uniform_location("u_direction");
    if (loc >= 0) blur_prog_->uniform2f(loc, horizontal ? 1.0f : 0.0f, horizontal ? 0.0f : 1.0f);

    int output_fbo = horizontal ? 1 : 0;
    bloom_fbo_[output_fbo]->bind();
    gl::viewport(0, 0, bw, bh);
    gl::clear(GL_COLOR_BUFFER_BIT);
    quad_vao_->bind();
    gl::draw_elements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
}

} // namespace gfx
