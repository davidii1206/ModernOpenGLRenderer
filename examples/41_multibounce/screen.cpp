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

namespace mbg {

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
    "Irradiance", "Outgoing", "Lit", "Irradiance (heat)",
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

} // namespace mbg
