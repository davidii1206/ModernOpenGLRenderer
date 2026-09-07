#pragma once

// ---------------------------------------------------------------------------
// Emitter proxies for next-event estimation.
//
// The microbuffer is both the visibility structure and the light sampling
// structure. That is right for indirect -- low frequency, large solid angles --
// and hopeless for direct: Cornell's ceiling panel lands on about 4 of 256
// buckets (finding 16), so the term that carries every sharp shadow is quantized
// to ~5 degrees and its penumbra is built from four samples of the light.
//
// So the direct term is split out and sampled explicitly, the way a path tracer
// does next event estimation. This file is the light side of that split: each
// emissive island is fitted with an oriented rectangle that the direct pass can
// stratify exactly.
//
// It is a HARD PARTITION of the emitter set, not two estimators of the same
// integral, so no MIS weights are needed: surfels belonging to a proxy still
// occlude in the microbuffer but contribute no radiance there. Example 39 does
// the same partition for its analytic path.
// ---------------------------------------------------------------------------

#include "surfels.hpp"

#include <gl/gl.hpp>
#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

namespace sgi {

// One area light, world space. 64 bytes; mirrored by shaders/common/bitmask.glsl.
//
//   p0 = centre.xyz, area
//   p1 = half_u.xyz, radiance.r
//   p2 = half_v.xyz, radiance.g
//   p3 = normal.xyz, radiance.b
//
// Corners are centre +/- half_u +/- half_v, so the shader rebuilds the rectangle
// without a separate vertex list.
struct GpuEmitter {
    glm::vec4 p0{0.0f};
    glm::vec4 p1{0.0f};
    glm::vec4 p2{0.0f};
    glm::vec4 p3{0.0f};
};
static_assert(sizeof(GpuEmitter) == 64, "GpuEmitter must stay 64 bytes");

enum EmitterBinding : uint32_t {
    kBindEmitters = 19,   // GpuEmitter[]
};

class EmitterSet {
public:
    // Groups emissive triangles into coplanar islands and fits one rectangle to
    // each. Cornell has a single island of two triangles.
    void build(const std::vector<Tri>& tris);

    void bind() const;
    uint32_t count() const { return uint32_t(emitters_.size()); }
    const std::vector<GpuEmitter>& emitters() const { return emitters_; }

    // Summed emissive triangle area, for the fit assertion and for the log.
    double source_area() const { return source_area_; }
    double proxy_area() const { return proxy_area_; }
    float  worst_plane_dev() const { return worst_plane_dev_; }

private:
    std::vector<GpuEmitter> emitters_;
    double source_area_ = 0.0, proxy_area_ = 0.0;
    float  worst_plane_dev_ = 0.0f;
    gl::Buffer b_{gl::BufferType::shader, gl::BufferUsage::static_draw};
};

} // namespace sgi
