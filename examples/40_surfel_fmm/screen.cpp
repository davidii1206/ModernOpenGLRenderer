#include "screen.hpp"

#include <gllib/log.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>

#ifndef GL_TEXTURE_SRGB_DECODE_EXT
#define GL_TEXTURE_SRGB_DECODE_EXT 0x8A48
#endif
#ifndef GL_SKIP_DECODE_EXT
#define GL_SKIP_DECODE_EXT 0x8A4A
#endif

namespace sgi {

// --- GBuffer ----------------------------------------------------------------

void GBuffer::create(int w, int h) {
    if (w == width && h == height) return;
    width = w;
    height = h;

    // gl::Texture::image_2d allocates IMMUTABLE storage, so re-specing an
    // existing texture is INVALID_OPERATION and silently keeps the old size.
    // Build fresh ones and move-assign.
    auto make = [&](gl::Texture& dst, GLint internal, GLenum fmt, GLenum type) {
        gl::Texture t(gl::TextureType::tex_2d);
        t.image_2d(0, internal, w, h, fmt, type, nullptr);
        t.parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        t.parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        t.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        t.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        dst = std::move(t);
    };

    make(albedo,   GL_RGBA8,              GL_RGBA,            GL_UNSIGNED_BYTE);
    make(normal,   GL_RGBA16F,            GL_RGBA,            GL_FLOAT);
    make(emissive, GL_RGBA16F,            GL_RGBA,            GL_FLOAT);
    make(depth,    GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT);

    gl::Framebuffer f;
    f.attach_texture(GL_COLOR_ATTACHMENT0, albedo);
    f.attach_texture(GL_COLOR_ATTACHMENT1, normal);
    f.attach_texture(GL_COLOR_ATTACHMENT2, emissive);
    f.attach_texture(GL_DEPTH_ATTACHMENT,  depth);
    fbo = std::move(f);

    fbo.bind();
    const GLenum bufs[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
    glDrawBuffers(3, bufs);
    if (!fbo.check()) gllib::log(gllib::LogLevel::error, "G-buffer framebuffer incomplete");
    gl::Framebuffer::unbind(gl::FramebufferType::both);
}

void GBuffer::bind_for_geometry() const {
    fbo.bind();
    gl::viewport(0, 0, width, height);
}

void GBuffer::bind_textures() const {
    albedo.bind(0);
    normal.bind(1);
    emissive.bind(2);
    depth.bind(3);
}

// --- GeometryPass -----------------------------------------------------------

bool GeometryPass::init() {
    prog_ = Pipeline::raster("shaders/gbuf.vert", "shaders/gbuf.frag");
    return prog_.valid();
}

void GeometryPass::render(const GBuffer& gb, const gfx::Model& model,
                          const glm::mat4& view_proj) {
    if (!prog_.valid()) return;

    gb.bind_for_geometry();
    gl::clear_color(0.0f, 0.0f, 0.0f, 0.0f);
    gl::enable(GL_DEPTH_TEST);
    gl::depth_func(GL_LESS);
    gl::depth_mask(GL_TRUE);
    gl::clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    prog_.use();
    prog_.set("u_view_proj", view_proj);

    for (std::size_t i = 0; i < model.mesh_count(); ++i) {
        const glm::mat4 xf = model.mesh_transform(i);
        prog_.set("u_model", xf);
        prog_.set("u_normal_mat", glm::mat3(glm::transpose(glm::inverse(glm::mat3(xf)))));

        glm::vec4 base(1.0f);
        glm::vec3 emi(0.0f);
        float rough = 1.0f, metal = 0.0f;
        bool two_sided = false;
        const int mat = model.mesh_material(i);
        if (mat >= 0 && std::size_t(mat) < model.material_count()) {
            const gfx::ModelMaterialInfo& m = model.material_info(std::size_t(mat));
            base  = glm::make_vec4(m.base_color_factor);
            emi   = glm::make_vec3(m.emissive_factor);
            rough = m.roughness_factor;
            metal = m.metallic_factor;
            two_sided = m.double_sided;
        }
        prog_.set("u_base_color_factor", base);
        prog_.set("u_emissive_factor", emi);
        prog_.set("u_roughness_factor", rough);
        prog_.set("u_metallic_factor", metal);

        if (two_sided) gl::disable(GL_CULL_FACE);
        else { gl::enable(GL_CULL_FACE); gl::cull_face(GL_BACK); }

        model.mesh(i).draw();
    }

    gl::disable(GL_CULL_FACE);
    gl::Framebuffer::unbind(gl::FramebufferType::both);
}

// --- SurfelPointsPass -------------------------------------------------------

namespace {
const char* const kColorModes[] = {
    "Albedo", "Normal", "Radius", "Emission", "Irradiance", "Outgoing", "Index",
};
}

const char* const* SurfelPointsPass::color_mode_names(int& count) {
    count = int(sizeof(kColorModes) / sizeof(kColorModes[0]));
    return kColorModes;
}

bool SurfelFilterPass::init() {
    prog_ = Pipeline::compute("shaders/surfel_filter.comp");
    grad_ = Pipeline::compute("shaders/light_grad.comp");
    return prog_.valid() && grad_.valid();
}

void SurfelFilterPass::ensure(uint32_t count) {
    if (count == count_) return;
    count_ = count;
    const std::vector<glm::vec4> zero(count, glm::vec4(0.0f));
    a_.data(zero.data(), zero.size() * sizeof(glm::vec4));
    b_.data(zero.data(), zero.size() * sizeof(glm::vec4));
    grad_buf_.data(zero.data(), zero.size() * sizeof(glm::vec4));
}

gl::Buffer* SurfelFilterPass::run(SurfelSet& set, const SurfelGrid& grid, int iterations,
                                  float radius, float plane_tol, float normal_tol,
                                  float light_sigma, float grad_scale, uint64_t version) {
    if (!prog_.valid() || set.count() == 0 || !grid.valid() || iterations <= 0) {
        cached_version_ = ~0ull;
        return nullptr;
    }
    // Nothing about the input changed, so neither did the output.
    if (version == cached_version_ && iterations == cached_iters_ &&
        radius == cached_radius_ && light_sigma == cached_sigma_ &&
        grad_scale == cached_grad_ &&
        cached_result_ != nullptr)
        return cached_result_;
    ensure(set.count());

    set.bind();
    grid.bind();
    grad_buf_.bind_base(kBindLightGrad);

    // Fit the local linear model of light_vis first: the filter's edge stop tests
    // the residual against it, which is what lets a penumbra ramp be denoised
    // instead of frozen.
    grad_.use();
    grad_.set("u_count", set.count());
    grad_.set("u_grid_min", grid.min());
    grad_.set("u_grid_res", grid.res());
    grad_.set("u_inv_cell", grid.inv_cell());
    grad_.set("u_radius", radius * set.spacing());
    grad_.set("u_plane_tol", plane_tol * set.spacing());
    grad_.set("u_normal_tol", normal_tol);
    grad_.set("u_light_sigma", light_sigma);
    gl::dispatch_compute((set.count() + 127u) / 128u, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    prog_.use();
    prog_.set("u_count", set.count());
    prog_.set("u_grid_min", grid.min());
    prog_.set("u_grid_res", grid.res());
    prog_.set("u_inv_cell", grid.inv_cell());
    prog_.set("u_radius", radius * set.spacing());
    prog_.set("u_plane_tol", plane_tol * set.spacing());
    prog_.set("u_normal_tol", normal_tol);
    prog_.set("u_light_sigma", light_sigma);
    prog_.set("u_grad_scale", grad_scale);

    // The shader reads kBindIrrad and writes kBindIrradFilt. Iteration 1 reads
    // the set's own buffer; later iterations rebind the previous result over
    // kBindIrrad, so the ping-pong needs no second code path in the shader.
    gl::Buffer* src = nullptr;
    gl::Buffer* dst = &a_;
    for (int it = 0; it < iterations; ++it) {
        if (src) src->bind_base(kBindIrrad);
        dst->bind_base(kBindIrradFilt);
        gl::dispatch_compute((set.count() + 127u) / 128u, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        src = dst;
        dst = (dst == &a_) ? &b_ : &a_;
    }
    cached_version_ = version;
    cached_iters_ = iterations;
    cached_radius_ = radius;
    cached_sigma_ = light_sigma;
    cached_grad_ = grad_scale;
    cached_result_ = src;
    return src;
}

bool SurfelGatherPass::init() {
    prog_ = Pipeline::compute("shaders/surfel_gather.comp");
    return prog_.valid();
}

void SurfelGatherPass::resize(int w, int h) {
    if (w == width_ && h == height_) return;
    width_ = w; height_ = h;
    // A plain RGBA32F target so the display pass cannot tell reconstructions
    // apart. The gather writes (E, 1) rather than (E*w, w), and display.frag's
    // divide by .a is then the identity.
    gl::Texture t(gl::TextureType::tex_2d);
    t.image_2d(0, GL_RGBA32F, w, h, GL_RGBA, GL_FLOAT, nullptr);
    t.parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    t.parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    t.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    t.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    accum_ = std::move(t);

    gl::Texture b(gl::TextureType::tex_2d);
    b.image_2d(0, GL_RG16F, w, h, GL_RG, GL_FLOAT, nullptr);
    b.parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    b.parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    b.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    b.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    bracket_ = std::move(b);
}

void SurfelGatherPass::render(const GBuffer& gb, SurfelSet& set, const SurfelGrid& grid,
                              const gfx::Camera& cam, float radius, float plane_tol,
                              float normal_tol, int kernel, float light_sigma,
                              float grad_scale, bool show_light, int mls, bool debug_fallback,
                              gl::Buffer* irrad) {
    if (!prog_.valid() || set.count() == 0 || !grid.valid()) return;

    set.bind();
    grid.bind();
    if (irrad) irrad->bind_base(kBindIrrad);
    gb.normal.bind(1);
    gb.depth.bind(3);
    accum_.bind_image(0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    bracket_.bind_image(2, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG16F);

    const glm::mat4 vp = cam.view_projection();
    prog_.use();
    prog_.set("u_inv_view_proj", glm::inverse(vp));
    prog_.set("u_viewport", glm::ivec2(width_, height_));
    prog_.set("u_grid_min", grid.min());
    prog_.set("u_grid_res", grid.res());
    prog_.set("u_inv_cell", grid.inv_cell());
    prog_.set("u_radius", radius * set.spacing());
    prog_.set("u_plane_tol", plane_tol * set.spacing());
    prog_.set("u_normal_tol", normal_tol);
    prog_.set("u_kernel", uint32_t(std::max(0, kernel)));
    prog_.set("u_light_sigma", light_sigma);
    prog_.set("u_grad_scale", grad_scale);
    prog_.set("u_show_light", show_light ? 1u : 0u);
    prog_.set("u_mls", uint32_t(std::max(0, mls)));
    prog_.set("u_debug", debug_fallback ? 1u : 0u);

    gl::dispatch_compute(uint32_t((width_ + 7) / 8), uint32_t((height_ + 7) / 8), 1);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

bool DirectPixelPass::init() {
    prog_ = Pipeline::compute("shaders/nee_pixel.comp");
    return prog_.valid();
}

void DirectPixelPass::render(const GBuffer& gb, SurfelSet& set, const SurfelGrid& grid,
                             const EmitterSet& emitters, const CutSet& cuts,
                             const gfx::Camera& cam,
                             const gl::Texture& target, const gl::Texture& bracket,
                             int width, int height,
                             const SolveConfig& cfg, bool show_light) {
    if (!prog_.valid() || set.count() == 0 || !grid.valid() || emitters.count() == 0) return;

    set.bind();
    grid.bind();
    emitters.bind();
    cuts.bind();
    gb.normal.bind(1);
    gb.depth.bind(3);
    // READ_WRITE, not WRITE_ONLY: the pass composites onto what the
    // reconstruction left behind.
    target.bind_image(0, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
    bracket.bind_image(2, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG16F);

    prog_.use();
    prog_.set("u_inv_view_proj", glm::inverse(cam.view_projection()));
    prog_.set("u_viewport", glm::ivec2(width, height));
    prog_.set("u_emitters", emitters.count());
    prog_.set("u_grid_min", grid.min());
    prog_.set("u_grid_res", grid.res());
    prog_.set("u_macro_res", grid.macro_res());
    prog_.set("u_inv_cell", grid.inv_cell());
    prog_.set("u_cell", grid.cell());
    prog_.set("u_thick", cfg.nee_thick);
    prog_.set("u_occ", cfg.nee_occ);
    prog_.set("u_cuts", (cfg.nee_cuts && cuts.valid()) ? 1u : 0u);
    prog_.set("u_bias", cfg.nee_bias);
    prog_.set("u_self_cos", cfg.nee_self_cos);
    prog_.set("u_self_tol", cfg.nee_self_tol);
    prog_.set("u_radius", set.radius());
    prog_.set("u_show_light", show_light ? 1u : 0u);
    prog_.set("u_skip", cfg.nee_skip);

    gl::dispatch_compute(uint32_t((width + 7) / 8), uint32_t((height + 7) / 8), 1);
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

bool SurfelPointsPass::init() {
    prog_ = Pipeline::raster("shaders/surfel_points.vert", "shaders/surfel_points.frag");
    if (empty_vao_ == 0) glCreateVertexArrays(1, &empty_vao_);
    return prog_.valid();
}

void SurfelPointsPass::render(const GBuffer& gb, SurfelSet& set, const glm::mat4& view_proj,
                              const gfx::Camera& cam, int color_mode, float point_scale,
                              float irradiance_gain) {
    if (!prog_.valid() || set.count() == 0) return;

    gl::Framebuffer::unbind(gl::FramebufferType::both);
    gl::viewport(0, 0, gb.width, gb.height);
    gl::disable(GL_DEPTH_TEST);
    gl::depth_mask(GL_FALSE);
    glEnable(GL_PROGRAM_POINT_SIZE);

    set.bind();
    gb.depth.bind(3);

    prog_.use();
    prog_.set("u_view_proj", view_proj);
    prog_.set("u_count", set.count());
    prog_.set("u_color_mode", color_mode);
    prog_.set("u_point_scale", point_scale);
    prog_.set("u_irradiance_gain", irradiance_gain);
    prog_.set("u_viewport", glm::vec2(float(gb.width), float(gb.height)));
    prog_.set("u_proj_scale", float(gb.height) * 0.5f /
                              std::tan(glm::radians(cam.fov()) * 0.5f));
    prog_.set("u_near", cam.near_clip());
    prog_.set("u_far", cam.far_clip());

    glBindVertexArray(empty_vao_);
    gl::draw_arrays(GL_POINTS, 0, GLsizei(set.count()));
    glBindVertexArray(0);

    glDisable(GL_PROGRAM_POINT_SIZE);
}

// --- References -------------------------------------------------------------

void References::load() {
    auto skip_srgb = [](gfx::Texture& t) {
        // gfx::Texture::load allocates GL_SRGB8_ALPHA8, so sampling would
        // hardware-decode to linear. The references are already tonemapped, so
        // we want the stored display-space bytes back.
        glTextureParameteri(t.handle(), GL_TEXTURE_SRGB_DECODE_EXT, GL_SKIP_DECODE_EXT);
        glTextureParameteri(t.handle(), GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTextureParameteri(t.handle(), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(t.handle(), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    };

    have_direct = direct.load("CornellBoxGroundTruthDirectLighting.png");
    if (have_direct) skip_srgb(direct);
    have_full = full.load("CornellBoxOriginalGroundTruth.png");
    if (have_full) skip_srgb(full);

    gllib::logf(gllib::LogLevel::info, "references: direct %s, full %s",
                have_direct ? "loaded" : "MISSING", have_full ? "loaded" : "MISSING");
}

// --- DisplayPass ------------------------------------------------------------

namespace {
const char* const kViewModes[] = {
    "Albedo", "Normal", "Emissive", "Depth", "World position",
    "Irradiance", "Outgoing", "Lit", "Gather fallback",
    "Reference", "Split", "Diff",
};
}

const char* const* DisplayPass::view_mode_names(int& count) {
    count = int(sizeof(kViewModes) / sizeof(kViewModes[0]));
    return kViewModes;
}

bool DisplayPass::init() {
    prog_ = Pipeline::raster("shaders/display.vert", "shaders/display.frag");
    // The fullscreen triangle comes from gl_VertexID, but core profile still
    // requires a bound VAO for the draw to be valid.
    if (empty_vao_ == 0) glCreateVertexArrays(1, &empty_vao_);
    if (dummy_ == 0) {
        glCreateTextures(GL_TEXTURE_2D, 1, &dummy_);
        glTextureStorage2D(dummy_, 1, GL_RGBA8, 1, 1);
        const uint32_t black = 0xFF000000u;
        glTextureSubImage2D(dummy_, 0, 0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, &black);
        glTextureParameteri(dummy_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(dummy_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    return prog_.valid();
}

void DisplayPass::render(const GBuffer& gb, const gl::Texture& recon,
                         const References& refs, const Params& p) {
    if (!prog_.valid()) return;

    gl::Framebuffer::unbind(gl::FramebufferType::both);
    gl::viewport(0, 0, gb.width, gb.height);
    gl::disable(GL_DEPTH_TEST);
    gl::depth_mask(GL_FALSE);

    gb.bind_textures();
    recon.bind(4);
    glBindTextureUnit(5, refs.have_direct ? refs.direct.handle() : dummy_);
    glBindTextureUnit(6, refs.have_full ? refs.full.handle() : dummy_);

    prog_.use();
    prog_.set("u_view_mode", p.view_mode);
    prog_.set("u_exposure", p.exposure);
    prog_.set("u_irradiance_gain", p.irradiance_gain);
    prog_.set("u_diff_gain", p.diff_gain);
    prog_.set("u_split_x", p.split_x);
    prog_.set("u_gt_index", p.gt_index);
    prog_.set("u_tonemap", p.tonemap);
    prog_.set("u_inv_view_proj", p.inv_view_proj);
    prog_.set("u_scene_min", p.scene_min);
    prog_.set("u_scene_extent", p.scene_extent);

    glBindVertexArray(empty_vao_);
    gl::draw_arrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
}

} // namespace sgi
