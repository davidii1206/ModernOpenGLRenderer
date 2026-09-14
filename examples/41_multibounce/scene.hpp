#pragma once

// ---------------------------------------------------------------------------
// The scene, as the secondary cameras see it: a flat world-space triangle soup
// with its material resolved, and the per-texel quadrature table for a
// hemi-octahedral target.
//
// Flat and unaccelerated is the point. Design doc section 5 ("Micro-triangles")
// is the engineering core of the technique -- cluster DAG LOD so that triangles
// land at 1-2 pixels in a tiny target -- and section 8.2 orders the work so that
// LOD is milestone 4, asserted against milestones 1 and 3. This file is what
// milestones 1 and 3 are made of: the full-resolution mesh, every triangle
// visited by every camera.
//
// UNITS, stated once because the doc's own section 4.4 and example 40's
// section 1.1 both warn that getting this wrong is invisible and consistent:
//
//   emission   emitted RADIANCE   L_e   [W/m^2/sr]
//   irradiance IRRADIANCE         E     [W/m^2]
//   outgoing   L_out = L_e + albedo * E / PI
//
// The /PI belongs to the EMITTING surface and appears exactly twice in this
// example: shaders/gather.comp (transport) and shaders/common/brdf.glsl
// (display). Nowhere else.
// ---------------------------------------------------------------------------

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>
#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

namespace mbg {

struct Bounds {
    glm::vec3 mn{ 1e30f};
    glm::vec3 mx{-1e30f};

    void add(const glm::vec3& p) { mn = glm::min(mn, p); mx = glm::max(mx, p); }
    bool valid() const { return mn.x <= mx.x; }
    glm::vec3 center() const { return 0.5f * (mn + mx); }
    glm::vec3 extent() const { return mx - mn; }
    float diagonal() const { return valid() ? glm::length(mx - mn) : 1.0f; }
    float radius() const { return 0.5f * diagonal(); }
};

// A world-space triangle with its material already resolved. Cornell is 32 flat
// untextured triangles, so a face normal and two constant colours are the whole
// of it -- no UVs, no texture read-back. Same extraction as example 40, so the
// two examples are shading the identical geometry.
struct Tri {
    glm::vec3 p[3]{};
    glm::vec3 n{0.0f, 1.0f, 0.0f};
    glm::vec3 albedo{0.0f};
    glm::vec3 emission{0.0f};      // emitted RADIANCE, strength already folded in
    float     area = 0.0f;
    bool      double_sided = false;
};

std::vector<Tri> extract_triangles(const gfx::Model& model);
double total_area(const std::vector<Tri>& tris);

// SSBO binding points, global to the example. See shaders/common/scene.glsl.
enum Binding : uint32_t {
    kBindTris     = 0,
    kBindCams     = 1,
    kBindQuad     = 2,
    kBindIrrad    = 3,
    kBindChildCam = 4,
    kBindChildW   = 5,
    kBindChildE   = 6,
    kBindVisOut   = 7,     // raster.comp's optional visibility dump, gates only
    kBindEmitters = 8,     // triangle indices of the analytic emitters
    kBindDirect   = 9,     // this level's (direct irradiance, visible fraction)
    kBindChildD   = 10,    // the next level's, read by the gather
    kBindClusters = 11,    // triangle-cluster bounds, for culling
    kBindGroups   = 12,    // bounds over RUNS of clusters: the coarse level
    kBindTriShade = 13,    // albedo and emission, read only for a winner
    kBindCounters = 14,    // traversal work counters; MBG_COUNT only
    kBindPerf     = 15,    // in-shader phase cycles; MBG_PERF only
};

// The GPU triangle, mirroring MbgTri in shaders/common/scene.glsl.
// A run of consecutive triangles and the box around them, which is the whole
// acceleration structure this example has. Fixed size rather than per submesh:
// a submesh is whatever the artist made it, and Sponza's floor is one enormous
// one whose box culls nothing, while a fixed run stays spatially tight because
// extract_triangles emits them in mesh order.
struct GpuCluster {
    glm::vec4 lo;          // w: index of the first triangle
    glm::vec4 hi;          // w: how many
};

// SPLIT, BECAUSE THE TRAVERSAL IS BANDWIDTH BOUND AND READS HALF OF IT.
//
// The inner loop fetches a triangle and asks whether a direction is inside it:
// about 155 operations against 96 bytes, or 1.6 flops per byte, where an RTX
// 3060 wants something nearer 36. It is nowhere close to compute bound -- and
// two of those six vec4s, the albedo and the emission, are never read by the
// traversal at all. They are shading data, wanted only once a winner is known.
//
// Interleaved they still cost bandwidth, because a cache line is 128 bytes and
// this stride is 96: a line carries 1.3 triangles and a third of what it carries
// is thrown away. Split, the geometry array strides 64 and a line carries two
// triangles with nothing wasted. Total memory is unchanged -- the same six vec4s
// live in two arrays instead of one.
//
// The plane normal stays with the positions rather than with the shading data,
// because lv_tri_hit's half-space rule (finding 19) needs its SIGN -- the
// outward direction, oriented against the shading normal when the winding
// disagreed -- so it cannot be recovered from the positions alone. The
// hemisphere could derive it from cross(p1-p0, p2-p0); the light view could not,
// so it is stored once and both read it.
struct GpuTriGeom {
    glm::vec4 p0, p1, p2;
    glm::vec4 n;           // w != 0: double-sided
};

struct GpuTriShade {
    glm::vec4 albedo;
    glm::vec4 emission;
};

class Scene {
public:
    // Returns false if the set exceeds the 16-bit triangle index the packed
    // depth key allows (see hemi.glsl). That cap is a property of this
    // example's 32-bit key, not of the method.
    bool build(const std::vector<Tri>& tris);
    void bind() const {
        buf_.bind_base(kBindTris);
        shade_.bind_base(kBindTriShade);
        emit_.bind_base(kBindEmitters);
        clusters_.bind_base(kBindClusters);
        groups_.bind_base(kBindGroups);
    }

    uint32_t count() const { return count_; }
    const Bounds& bounds() const { return bounds_; }
    double area() const { return area_; }
    uint32_t emissive_count() const { return emissive_; }
    // Triangles whose direct contribution is evaluated analytically rather than
    // being picked up by the hemisphere raster. See raster.comp.
    uint32_t emitter_count() const { return emitter_count_; }
    uint32_t cluster_count() const { return cluster_count_; }
    uint32_t group_count() const { return group_count_; }
    // Clusters per group. Two levels of 64 cover 4096 clusters -- 262k
    // triangles -- in sqrt(n) tests instead of n, which is the whole point.
    static constexpr uint32_t kGroupSize = 64;
    // Triangles per cluster. 64 matches the workgroup, so one cull step feeds
    // one thread-per-triangle pass, and it is small enough that a cluster's box
    // is tight even where the mesh is not.
    static constexpr uint32_t kClusterSize = 64;

    static constexpr uint32_t kMaxTris = 0xFFFFFFFEu;   // MBG_EMPTY is the sentinel

private:
    gl::Buffer buf_{gl::BufferType::shader, gl::BufferUsage::static_draw};
    gl::Buffer shade_{gl::BufferType::shader, gl::BufferUsage::static_draw};
    gl::Buffer emit_{gl::BufferType::shader, gl::BufferUsage::static_draw};
    gl::Buffer clusters_{gl::BufferType::shader, gl::BufferUsage::static_draw};
    gl::Buffer groups_{gl::BufferType::shader, gl::BufferUsage::static_draw};
    uint32_t count_ = 0, emissive_ = 0, emitter_count_ = 0, cluster_count_ = 0,
             group_count_ = 0;
    double area_ = 0.0;
    Bounds bounds_;
};

// --- Quadrature -------------------------------------------------------------
//
// texel -> (unit direction in the tangent frame, INTEGRAL of cos(theta) dOmega
// over that texel).
//
// Doc section 4.4 says "weight by cosine and octahedral solid angle" and leaves
// it there; example 40 found the same sentence in its own spec and found that
// taking it literally -- one direction and a flat 2*PI/n solid angle per texel --
// costs 1.7-3.1% of total energy, worsening with resolution. The mapping's
// Jacobian is
//
//     dOmega = (1/2) / |w|^3  de     with  w(e) unnormalized, |w| in [1/sqrt2, 1]
//
// (the triple product w . (dw/de_x x dw/de_y) is identically 1/2 here, which is
// the cheapest way to get this right and the easiest to get wrong: the CROSS
// PRODUCT MAGNITUDE alone is sqrt(3)/2 and gives an answer sqrt(3) too large).
//
// Each texel's weight is INTEGRATED over its square with a subsampled midpoint
// rule rather than point-sampled at the centre, and the table is then normalized
// so the sum is exactly PI. Both sums are reported at startup: they are the
// cheapest possible version of section 8.2 milestone 1's "validate the
// integration math before any performance work".
struct Quadrature {
    uint32_t res = 0;
    std::vector<glm::vec4> texels;
    double sum_omega = 0.0;        // raw, before normalization; should be 2*PI
    double sum_cos = 0.0;          // raw, before normalization; should be PI
    gl::Buffer buf{gl::BufferType::shader, gl::BufferUsage::static_draw};

    void build(uint32_t res, uint32_t subsamples = 8);
    void bind() const { buf.bind_base(kBindQuad); }
};

// Unnormalized hemi-octahedral direction for a point in the square [-1,1]^2.
// The CPU mirror of mbg_px_to_dir in shaders/common/hemi.glsl.
glm::vec3 hemi_oct_dir(glm::vec2 e);

} // namespace mbg
