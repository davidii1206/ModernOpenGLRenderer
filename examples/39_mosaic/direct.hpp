#pragma once

// ---------------------------------------------------------------------------
// M5 — the direct-lighting addendum: emitter proxies and analytic area lights.
//
// Up to M4 the only path from an emissive surface to a receiver ran through the
// gather: the emitter's surfels were splatted into every receiver's 16x16
// micro-buffer like any other geometry. That is correct in the limit and awful
// in practice for a SMALL BRIGHT source. A Cornell ceiling panel lands on a
// handful of texels out of 256, so its contribution is quantized to the
// micro-buffer's angular resolution, and once it is far enough to be aggregated
// it is averaged against the dark ceiling around it inside one cluster.
//
// The fix the spec prescribes is to represent such sources EXPLICITLY, as a
// small set of oriented rectangles, and shade them analytically. A rectangle's
// diffuse contribution has a closed form (the Lambert/Arvo polygon integral), so
// there is no sampling and therefore no noise or banding from the emitter at
// all -- only from its visibility.
//
// DEVIATION FROM THE SPEC. The spec extracts proxies by rasterizing emissive
// textures into UV space, running connected components there, and fitting rects
// per island. This does the connected components and the PCA fit over the
// EMISSIVE SURFELS THAT M1 ALREADY BAKED instead. They carry texture-sampled
// emissive radiance, they are already spatially distributed over the surface,
// and they cost nothing extra to produce. Same output -- centre, two half-axes,
// normal, area, mean radiance -- without writing a UV-space rasterizer.
//
// Proxied emission is REMOVED from the gather (see surfel_direct.comp), so the
// two paths partition the light rather than double-counting it.
// ---------------------------------------------------------------------------

#include "mosaic.hpp"
#include "surfel_bake.hpp"

#include <gl/gl.hpp>
#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

namespace mosaic {

// One area-light rectangle, world space. 64 bytes; the GLSL mirror is
// shaders/common/ltc.glsl.
//
//   p0 = centre.xyz,  area
//   p1 = half_u.xyz,  radiance.r
//   p2 = half_v.xyz,  radiance.g
//   p3 = normal.xyz,  radiance.b
//
// The corners are centre +/- half_u +/- half_v, so the shader reconstructs the
// polygon without a separate vertex list.
struct GpuEmitter {
    glm::vec4 p0{0.0f};
    glm::vec4 p1{0.0f};
    glm::vec4 p2{0.0f};
    glm::vec4 p3{0.0f};
};
static_assert(sizeof(GpuEmitter) == 64, "GpuEmitter must stay 64 bytes");

// Object-space proxy, fitted once per (model, mesh) surfel set and instanced.
struct EmitterProxy {
    glm::vec3 center{0.0f};
    glm::vec3 half_u{0.0f};
    glm::vec3 half_v{0.0f};
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
    glm::vec3 radiance{0.0f};
    float coverage = 0.0f;    // fitted surface area / rectangle area, a fit-quality score
    // Source-set indices this proxy accounts for, so S2 can drop their emission
    // from the gather. Per surfel and not per mesh: one mesh can have an island
    // that fits a rectangle and another that does not.
    std::vector<uint32_t> members;
};

// Fits proxies to one set's LOD-0 emissive surfels. `spacing` is that LOD's
// Poisson spacing in the set's own units and sets both the connectivity radius
// and the per-surfel area estimate.
std::vector<EmitterProxy> fit_emitter_proxies(const SurfelSet& set, float spacing);

// Builds the world-space emitter list for a frame's instances and uploads it.
class EmitterSet {
public:
    // Static geometry only: proxies are fitted once and instanced. Movers are
    // re-transformed per frame by rebuild().
    void build(SurfelLibrary& lib, const std::vector<Instance>& instances);

    // Re-transforms the object-space proxies with the instances' current
    // matrices. Cheap: a handful of rectangles, no refitting.
    void refresh(const std::vector<Instance>& instances);

    gl::Buffer& buffer() { return buffer_; }
    uint32_t count() const { return uint32_t(gpu_.size()); }
    const std::vector<GpuEmitter>& emitters() const { return gpu_; }

    // Marks every surfel set that contributed a proxy, so S2 can drop its
    // emission from the gather. Indexed by library entry.
    const std::vector<uint8_t>& proxied_entries() const { return proxied_; }
    gl::Buffer& proxied_buffer() { return proxied_buf_; }

    uint32_t fitted() const { return fitted_; }
    uint32_t rejected() const { return rejected_; }

private:
    struct Placed {
        int instance = 0;
        EmitterProxy proxy;
    };
    std::vector<Placed> placed_;
    std::vector<GpuEmitter> gpu_;
    std::vector<uint8_t> proxied_;
    gl::Buffer buffer_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    // One byte per SOURCE surfel: 1 when a proxy already accounts for its
    // emission. A per-entry flag is not enough -- a mesh can have one island
    // that fits a rectangle and another that does not.
    gl::Buffer proxied_buf_{gl::BufferType::shader, gl::BufferUsage::static_draw};
    uint32_t fitted_ = 0;
    uint32_t rejected_ = 0;
};

} // namespace mosaic
