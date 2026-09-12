#pragma once

// ---------------------------------------------------------------------------
// The screen-space side: the primary G-buffer and its geometry pass, the
// path-traced references, and the display/compare pass.
//
// Doc section 5.5's closing note: "keep the hardware path as a correctness
// oracle and for the primary G-buffer pass". That is exactly what this file is.
// The secondary cameras never touch it -- they are software-rasterized in
// shaders/raster.comp -- so the only hardware rasterization in the example is
// this one pass and the fullscreen triangle that displays the result.
//
// Adapted from example 40 with its surfel passes removed, deliberately keeping
// the identical G-buffer layout, tone curve and view modes so that a split view
// between the two examples compares estimators rather than presentation.
// ---------------------------------------------------------------------------

#include "gpu_util.hpp"

namespace mbg {

// Deferred G-buffer.
//
//   gbuf0  RGBA8   albedo.rgb | a: 1 = geometry, 0 = background
//   gbuf1  RGBA16F normal.xyz (world) | w: roughness
//   gbuf2  RGBA16F emissive.rgb | w: metallic
//   depth  DEPTH32F
//
// World position is reconstructed from depth, not stored. The camera placement
// pass (shaders/place.comp) reads exactly these four.
struct GBuffer {
    gl::Framebuffer fbo;
    gl::Texture albedo, normal, emissive, depth;
    int width = 0, height = 0;

    void create(int w, int h);
    void bind_for_geometry() const;
    void bind_textures() const;      // units 0..3
};

class GeometryPass {
public:
    bool init();
    bool poll() { return prog_.poll(); }
    void render(const GBuffer& gb, const gfx::Model& model, const glm::mat4& view_proj);
private:
    Pipeline prog_;
};

// Path-traced references, loaded from PNG and compared against in display space.
//
// Both are Blender Cycles renders of CornellBoxOriginal.glb at 512x512, copied
// from example 40 so the two examples are scored against the same images:
//   CornellBoxGroundTruthDirectLighting.png   direct only  == 1 camera level
//   CornellBoxOriginalGroundTruth.png         converged multi-bounce
struct References {
    gfx::Texture direct;
    gfx::Texture full;
    bool have_direct = false, have_full = false;
    void load();
};

class DisplayPass {
public:
    struct Params {
        int   view_mode = 7;
        float exposure = 1.0f;
        float irradiance_gain = 1.0f;
        float diff_gain = 4.0f;
        float split_x = 0.5f;
        int   gt_index = 1;            // 0 = direct reference, 1 = full-GI reference
        int   tonemap = 0;
        glm::mat4 inv_view_proj{1.0f};
        glm::vec3 scene_min{0.0f}, scene_extent{1.0f};
    };

    bool init();
    bool poll() { return prog_.poll(); }
    void render(const GBuffer& gb, const gl::Texture& recon,
                const References& refs, const Params& p);
    static const char* const* view_mode_names(int& count);

private:
    Pipeline prog_;
    GLuint empty_vao_ = 0;
    GLuint dummy_ = 0;
};

} // namespace mbg
