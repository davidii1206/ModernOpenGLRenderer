#include "screen.hpp"
#include "clipmap.hpp"
#include "cache_gi.hpp"   // bind_occupancy
#include "direct.hpp"

#include <gllib/log.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>

namespace mosaic {

// --- GBuffer ----------------------------------------------------------------

void GBuffer::create(int w, int h) {
    if (w == width && h == height) return;
    width = w;
    height = h;

    // Immutable storage: build fresh textures and move-assign. Re-specing an
    // existing gl::Texture is INVALID_OPERATION and keeps the old size, which
    // shows up much later as a mysteriously stale-resolution G-buffer.
    auto make = [&](gl::Texture& dst, GLint internal, GLenum fmt, GLenum type) {
        gl::Texture t(gl::TextureType::tex_2d);
        t.image_2d(0, internal, w, h, fmt, type, nullptr);
        t.parameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        t.parameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        t.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        t.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        dst = std::move(t);
    };

    make(albedo,   GL_RGBA8,             GL_RGBA,            GL_UNSIGNED_BYTE);
    make(normal,   GL_RGBA16F,           GL_RGBA,            GL_FLOAT);
    make(emissive, GL_RGBA16F,           GL_RGBA,            GL_FLOAT);
    make(motion,   GL_RG16F,             GL_RG,              GL_FLOAT);
    make(depth,    GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT);

    gl::Framebuffer f;
    f.attach_texture(GL_COLOR_ATTACHMENT0, albedo);
    f.attach_texture(GL_COLOR_ATTACHMENT1, normal);
    f.attach_texture(GL_COLOR_ATTACHMENT2, emissive);
    f.attach_texture(GL_COLOR_ATTACHMENT3, motion);
    f.attach_texture(GL_DEPTH_ATTACHMENT,  depth);
    fbo = std::move(f);

    fbo.bind();
    const GLenum bufs[4] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
                            GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3};
    glDrawBuffers(4, bufs);
    if (!fbo.check())
        gllib::log(gllib::LogLevel::error, "G-buffer framebuffer incomplete");
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
    motion.bind(3);
    depth.bind(4);
}

// --- GeometryPass -----------------------------------------------------------

bool GeometryPass::init() {
    prog_ = Pipeline::raster("shaders/gbuf.vert", "shaders/gbuf.frag");
    return prog_.valid();
}

void GeometryPass::render(const GBuffer& gb,
                          const std::vector<Instance>& instances,
                          const std::vector<const gfx::Model*>& models,
                          const glm::mat4& view_proj,
                          const glm::mat4& prev_view_proj) {
    if (!prog_.valid()) return;

    gb.bind_for_geometry();
    gl::clear_color(0.0f, 0.0f, 0.0f, 0.0f);
    gl::enable(GL_DEPTH_TEST);
    gl::depth_func(GL_LESS);
    gl::depth_mask(GL_TRUE);
    gl::clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    prog_.use();
    prog_.set("u_view_proj", view_proj);
    prog_.set("u_prev_view_proj", prev_view_proj);

    for (const Instance& inst : instances) {
        if (inst.model < 0 || size_t(inst.model) >= models.size()) continue;
        const gfx::Model& model = *models[size_t(inst.model)];
        if (size_t(inst.mesh) >= model.mesh_count()) continue;

        prog_.set("u_model", inst.xform);
        prog_.set("u_prev_model", inst.prev_xform);
        prog_.set("u_normal_mat", inst.normal_matrix());

        // Material defaults, overridden below when the instance has one.
        glm::vec4 base_factor(1.0f);
        glm::vec3 emissive_factor(0.0f);
        float metallic = 0.0f, roughness = 1.0f, alpha_cutoff = 0.5f;
        int has_base = 0, has_mr = 0, has_emi = 0, alpha_mask = 0;
        bool double_sided = false;

        if (inst.material >= 0 && size_t(inst.material) < model.material_count()) {
            const gfx::ModelMaterialInfo& m = model.material_info(size_t(inst.material));
            base_factor = glm::make_vec4(m.base_color_factor);
            emissive_factor = glm::make_vec3(m.emissive_factor);
            metallic = m.metallic_factor;
            roughness = m.roughness_factor;
            alpha_cutoff = m.alpha_cutoff;
            alpha_mask = (m.alpha_mode == gfx::AlphaMode_Mask) ? 1 : 0;
            double_sided = m.double_sided;

            // Texture indices are glTF IMAGE indices, and Model skips images it
            // failed to decode, so an out-of-range index is possible on a
            // damaged asset. Bounds-check rather than trusting the index.
            auto bind_tex = [&](int idx, int unit) -> int {
                if (idx < 0 || size_t(idx) >= model.texture_count()) return 0;
                const auto& t = model.texture(size_t(idx));
                if (!t) return 0;
                t->bind(unit);
                return 1;
            };
            has_base = bind_tex(m.base_color_tex, 0);
            has_mr   = bind_tex(m.metallic_roughness_tex, 1);
            has_emi  = bind_tex(m.emissive_tex, 2);
        }

        prog_.set("u_base_color_factor", base_factor);
        prog_.set("u_emissive_factor", emissive_factor);
        prog_.set("u_metallic_factor", metallic);
        prog_.set("u_roughness_factor", roughness);
        prog_.set("u_has_base_color", has_base);
        prog_.set("u_has_mr", has_mr);
        prog_.set("u_has_emissive", has_emi);
        prog_.set("u_alpha_cutoff", alpha_cutoff);
        prog_.set("u_alpha_mask", alpha_mask);

        if (double_sided) gl::disable(GL_CULL_FACE);
        else { gl::enable(GL_CULL_FACE); gl::cull_face(GL_BACK); }

        model.mesh(size_t(inst.mesh)).draw();
    }

    gl::disable(GL_CULL_FACE);
    gl::Framebuffer::unbind(gl::FramebufferType::both);
}

// --- DisplayPass ------------------------------------------------------------

namespace {
const char* const kViewModes[] = {
    "Albedo", "Normal", "Roughness", "Metallic",
    "Emissive", "Motion", "World position", "Depth",
    "Indirect", "Lit",
};
}

const char* const* DisplayPass::view_mode_names(int& count) {
    count = int(sizeof(kViewModes) / sizeof(kViewModes[0]));
    return kViewModes;
}

// 1x1 "always lit" depth texture, bound whenever there is no sun shadow map.
static GLuint make_dummy_shadow() {
    GLuint t = 0;
    glCreateTextures(GL_TEXTURE_2D, 1, &t);
    glTextureStorage2D(t, 1, GL_DEPTH_COMPONENT32F, 1, 1);
    const float one = 1.0f;
    glTextureSubImage2D(t, 0, 0, 0, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &one);
    glTextureParameteri(t, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(t, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(t, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(t, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureParameteri(t, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTextureParameteri(t, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    return t;
}

bool DisplayPass::init() {
    prog_ = Pipeline::raster("shaders/display.vert", "shaders/display.frag");
    // The fullscreen triangle is generated from gl_VertexID, but core profile
    // still requires a bound VAO for the draw to be valid.
    if (empty_vao_ == 0) glCreateVertexArrays(1, &empty_vao_);
    if (dummy_shadow_ == 0) dummy_shadow_ = make_dummy_shadow();
    return prog_.valid();
}

void DisplayPass::render(const GBuffer& gb, int view_mode, float exposure,
                         const glm::mat4& inv_view_proj,
                         const glm::vec3& scene_min, const glm::vec3& scene_extent,
                         const Lighting& light) {
    if (!prog_.valid()) return;

    gl::Framebuffer::unbind(gl::FramebufferType::both);
    gl::viewport(0, 0, gb.width, gb.height);
    gl::disable(GL_DEPTH_TEST);
    gl::depth_mask(GL_FALSE);

    gb.bind_textures();
    prog_.use();
    prog_.set("u_view_mode", view_mode);
    prog_.set("u_exposure", exposure);
    prog_.set("u_inv_view_proj", inv_view_proj);
    prog_.set("u_scene_min", scene_min);
    prog_.set("u_scene_extent", scene_extent);

    if (light.indirect) light.indirect->bind(5);
    // Unit 6 is a sampler2DShadow. Leaving a stale colour texture there is
    // undefined behaviour even when the u_have_sun branch never samples it, so
    // an unlit frame gets a 1x1 depth texture instead.
    glBindTextureUnit(6, light.sun_shadow ? light.sun_shadow : dummy_shadow_);
    prog_.set("u_sun_dir", glm::normalize(light.sun_dir));
    prog_.set("u_sun_radiance", light.sun_radiance);
    prog_.set("u_sun_view_proj", light.sun_view_proj);
    prog_.set("u_shadow_texel", light.shadow_texel);
    prog_.set("u_have_sun", (light.have_sun && light.sun_shadow) ? 1 : 0);
    prog_.set("u_emissive_boost", light.emissive_boost);
    prog_.set("u_sky", light.sky);
    prog_.set("u_cam_pos", light.cam_pos);

    // Analytic area lights plus the occupancy field their cone march reads.
    if (light.emitters && light.clip) {
        light.emitters->buffer().bind_base(0);
        prog_.set("u_emitter_count", light.emitters->count());
        prog_.set("u_emitter_steps", light.emitter_steps);
        bind_occupancy(prog_, *light.clip, 7);
    } else {
        prog_.set("u_emitter_count", 0u);
    }
    prog_.set("u_indirect_gain", light.indirect_gain);

    glBindVertexArray(empty_vao_);
    gl::draw_arrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    gl::depth_mask(GL_TRUE);
    gl::enable(GL_DEPTH_TEST);
}

// --- ScreenGI ---------------------------------------------------------------

const char* ScreenGI::subpass_name(int i) {
    static const char* names[] = {"S7 gather", "S7 upsample"};
    return names[i];
}

bool ScreenGI::init() {
    for (int i = 0; i < kSubPassCount; ++i)
        sub_[i] = std::make_unique<PassTimer>(subpass_name(i));
    gather_ = Pipeline::compute("shaders/screen_gather.comp");
    upsample_ = Pipeline::compute("shaders/upsample.comp");
    return gather_.valid() && upsample_.valid();
}

bool ScreenGI::poll() {
    bool c = gather_.poll();
    c |= upsample_.poll();
    return c;
}

void ScreenGI::resize(int w, int h) {
    if (w == width_ && h == height_) return;
    width_ = w;
    height_ = h;
    // Immutable storage: build fresh and move-assign (see GBuffer::create).
    auto make = [](gl::Texture& dst, int tw, int th) {
        gl::Texture t(gl::TextureType::tex_2d);
        t.image_2d(0, GL_RGBA16F, tw, th, GL_RGBA, GL_FLOAT, nullptr);
        t.parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        t.parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        t.parameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        t.parameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        dst = std::move(t);
    };
    make(half_, std::max(1, w / 2), std::max(1, h / 2));
    make(full_, w, h);
}

void ScreenGI::render(Clipmap& clip, const GBuffer& gb, const gfx::Camera& cam,
                      gl::Buffer& sh_cache, const glm::vec3& sky, int search_radius,
                      float reach) {
    if (!gather_.valid() || width_ == 0) return;
    const glm::ivec2 half(std::max(1, width_ / 2), std::max(1, height_ / 2));
    const glm::ivec2 full(width_, height_);

    sub_[kGather]->begin();
    gb.normal.bind(0);
    gb.depth.bind(1);
    clip.live_pn().bind_base(0);
    clip.live_src().bind_base(1);
    sh_cache.bind_base(2);
    clip.cell_sc().bind_base(3);
    clip.packed_idx().bind_base(4);
    clip.cluster_rad().bind_base(5);
    half_.bind_image(0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

    gather_.use();
    gather_.set("u_half_size", half);
    gather_.set("u_full_size", full);
    gather_.set("u_inv_view_proj", glm::inverse(cam.view_projection()));
    gather_.set("u_origin", clip.cascade(0).origin);
    gather_.set("u_inv_cell", 1.0f / clip.cascade(0).cell);
    gather_.set("u_cnt_off", clip.cascade(0).cnt_off);
    gather_.set("u_sky", sky);
    gather_.set("u_search_radius", search_radius);
    gather_.set("u_reach", reach);
    gather_.set("u_cascades", clip.active_cascades());
    for (int l = 0; l < clip.active_cascades(); ++l) {
        char name[32];
        std::snprintf(name, sizeof(name), "u_origins[%d]", l);
        gather_.set(name, clip.cascade(l).origin);
        std::snprintf(name, sizeof(name), "u_inv_cells[%d]", l);
        gather_.set(name, 1.0f / clip.cascade(l).cell);
        std::snprintf(name, sizeof(name), "u_cnt_offs[%d]", l);
        gather_.set(name, clip.cascade(l).cnt_off);
    }
    gl::dispatch_compute(GLuint((half.x + 7) / 8), GLuint((half.y + 7) / 8), 1);
    gl::memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
    sub_[kGather]->end();

    sub_[kUpsample]->begin();
    gb.normal.bind(0);
    gb.depth.bind(1);
    half_.bind(2);
    full_.bind_image(0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);
    upsample_.use();
    upsample_.set("u_full_size", full);
    upsample_.set("u_half_size", half);
    upsample_.set("u_near", cam.near_clip());
    upsample_.set("u_far", cam.far_clip());
    gl::dispatch_compute(GLuint((full.x + 7) / 8), GLuint((full.y + 7) / 8), 1);
    gl::memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
    sub_[kUpsample]->end();
}

// --- SurfelDebugPass --------------------------------------------------------

namespace {
const char* const kColorModes[] = {
    "Albedo", "Normal", "LOD", "Emissive", "Radius", "Two-sided",
    "Irradiance", "Albedo x irradiance", "Update age",
};
}

const char* const* SurfelDebugPass::color_mode_names(int& count) {
    count = int(sizeof(kColorModes) / sizeof(kColorModes[0]));
    return kColorModes;
}

bool SurfelDebugPass::init() {
    prog_ = Pipeline::raster("shaders/surfel_points.vert", "shaders/surfel_points.frag");
    if (empty_vao_ == 0) glCreateVertexArrays(1, &empty_vao_);
    return prog_.valid();
}

void SurfelDebugPass::render(SurfelLibrary& lib,
                             const std::vector<Instance>& instances,
                             const glm::mat4& view_proj,
                             const GBuffer& gb,
                             int lod, int color_mode, float point_scale,
                             float near_clip, float far_clip,
                             gl::Buffer* sh_cache, float irradiance_gain,
                             const std::vector<int>* live_lod) {
    if (!prog_.valid() || lib.total_surfels() == 0) return;

    // Two separate occlusion problems, solved separately:
    //
    //  - point vs SCENE: resolved in the fragment shader against the G-buffer
    //    depth texture. A depth blit is not an option because the G-buffer is
    //    DEPTH32F and the default framebuffer is not ("Depth formats do not
    //    match"), and copying into the G-buffer's own depth would corrupt it
    //    for the passes that read it.
    //
    //  - point vs POINT: the default framebuffer's own depth buffer, cleared
    //    here and written normally. Without this the cloud paints in buffer
    //    order and far surfels overwrite near ones, which reads as noise.
    gl::Framebuffer::unbind(gl::FramebufferType::both);
    gl::viewport(0, 0, gb.width, gb.height);
    gl::depth_mask(GL_TRUE);
    gl::clear(GL_DEPTH_BUFFER_BIT);
    gl::enable(GL_DEPTH_TEST);
    gl::depth_func(GL_LESS);
    gl::enable(GL_PROGRAM_POINT_SIZE);

    gb.depth.bind(4);

    lib.packed_buffer().bind_base(0);
    lib.parent_buffer().bind_base(1);
    if (sh_cache) sh_cache->bind_base(2);

    prog_.use();
    prog_.set("u_view_proj", view_proj);
    prog_.set("u_viewport", glm::vec2(float(gb.width), float(gb.height)));
    prog_.set("u_color_mode", color_mode);
    prog_.set("u_point_scale", point_scale);
    prog_.set("u_near", near_clip);
    prog_.set("u_far", far_clip);
    prog_.set("u_irradiance_gain", irradiance_gain);

    glBindVertexArray(empty_vao_);
    for (size_t ii = 0; ii < instances.size(); ++ii) {
        const Instance& inst = instances[ii];
        const int ei = lib.entry_for(inst);
        if (ei < 0) continue;
        const SurfelLibrary::Entry& e = lib.entries()[size_t(ei)];

        int want = lod;
        if (want < 0) {
            if (!live_lod || ii >= live_lod->size()) continue;
            want = (*live_lod)[ii];
            if (want < 0) continue;   // instance culled this frame
        }
        const int l = std::clamp(want, 0, Config::kSurfelLods - 1);
        const SurfelSet::Lod& L = e.set.lods[l];
        if (L.count == 0) continue;

        prog_.set("u_model", inst.xform);
        prog_.set("u_normal_mat", inst.normal_matrix());
        prog_.set("u_aabb_min", e.set.aabb_min);
        prog_.set("u_aabb_extent", e.set.aabb_max - e.set.aabb_min);
        prog_.set("u_radius_scale", e.set.radius_scale);
        prog_.set("u_base", e.base + L.offset);
        prog_.set("u_lod", unsigned(l));
        gl::draw_arrays(GL_POINTS, 0, GLsizei(L.count));
    }
    glBindVertexArray(0);

    gl::disable(GL_PROGRAM_POINT_SIZE);
    gl::depth_mask(GL_TRUE);
    gl::enable(GL_DEPTH_TEST);
}

} // namespace mosaic
