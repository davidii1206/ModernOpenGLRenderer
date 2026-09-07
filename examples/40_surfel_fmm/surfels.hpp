#pragma once

// ---------------------------------------------------------------------------
// M0b -- the surfel set.
//
// One static, world-space, SoA surfel set, laid out exactly as
// surfel-gi-spec-r3.md section 1.1 prescribes. There is no Poisson bake, no LOD
// pyramid, no disk cache and no object-space indirection here: this example's
// brute-force stage wants a surfel count it can sweep (O(N^2)) and an exact
// area invariant, not the machinery example 39 needs for Sponza.
//
// UNITS, stated once, because section 1.1 says getting this wrong is the most
// common failure in this method:
//
//   emission   is emitted RADIANCE  L_e  [W/m^2/sr]
//   irradiance is IRRADIANCE        E    [W/m^2]
//   outgoing   L_out = L_e + albedo * E / PI
//
// The /PI belongs to the EMITTING surfel and must appear everywhere L_out is
// formed. It is in bf_radiance.comp and in bf_micro.comp.
// ---------------------------------------------------------------------------

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>
#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

namespace sgi {

// --- Bounds -----------------------------------------------------------------

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

// --- Source geometry --------------------------------------------------------

// A world-space triangle with its material already resolved. Cornell is 32
// flat, untextured triangles, so a face normal and two constant colours are the
// whole of it -- no UVs, no texture read-back.
struct Tri {
    glm::vec3 p[3]{};
    glm::vec3 n{0.0f, 1.0f, 0.0f};   // face normal
    glm::vec3 albedo{0.0f};
    glm::vec3 emission{0.0f};        // emitted RADIANCE, strength already folded in
    float     area = 0.0f;
    bool      double_sided = false;
};

// Applies model.mesh_transform(i) and nothing else. CornellBoxOriginal.glb's
// single node carries a +90 degree X quaternion that takes the Z-up model to
// Y-up, so the extra rotate() example 36 applies would double it.
std::vector<Tri> extract_triangles(const gfx::Model& model);

double total_area(const std::vector<Tri>& tris);

// --- Packing, mirrored by shaders/common/surfel.glsl ------------------------

uint32_t  pack_oct_snorm16(const glm::vec3& n);
glm::vec3 unpack_oct_snorm16(uint32_t p);
uint32_t  pack_rgb8(const glm::vec3& c, uint32_t flags);
uint32_t  pack_rgb9e5(const glm::vec3& c);
glm::vec3 unpack_rgb9e5(uint32_t v);

enum SurfelFlags : uint32_t {
    kSurfelEmissive     = 1u << 0,
    kSurfelDoubleSided  = 1u << 1,
};

// SSBO binding points. Every compute shader in this example binds the set with
// SurfelSet::bind(), so these numbers are global to the example rather than
// per-pass.
enum SurfelBinding : uint32_t {
    kBindPosRad    = 0,   // vec4  pos.xyz, radius
    kBindNormal    = 1,   // uint  octahedral snorm16
    kBindAlbedo    = 2,   // uint  RGB8 + 8b flags
    kBindEmission  = 3,   // uint  RGB9E5 emitted radiance
    kBindIrrad     = 4,   // vec4  irradiance E, written this sweep
    kBindIrradPrev = 5,   // vec4  irradiance E, previous sweep (read)
};

// --- The set ----------------------------------------------------------------

class SurfelSet {
public:
    // Stratified area-weighted sampling. `target_count` is a target, not a
    // guarantee: every triangle gets at least one surfel, so the realised count
    // can exceed it slightly on a mesh with many tiny faces.
    void build(const std::vector<Tri>& tris, uint32_t target_count, uint32_t seed = 12345u);

    // Hand-built set, for the section 9 analytic gates.
    void build_explicit(const std::vector<glm::vec4>& pos_rad,
                        const std::vector<glm::vec3>& normals,
                        const std::vector<glm::vec3>& albedos,
                        const std::vector<glm::vec3>& emissions);

    // Zeroes both irradiance buffers. Every solve starts here.
    void reset_irradiance();

    // Binds all six buffers to kBind*.
    void bind() const;

    // Ping-pong at a SWEEP boundary. Section 7 forbids ping-ponging, but only
    // because it updates a subset of receivers per frame and a skipped receiver
    // would then read two-sweep-old data. Here every receiver is covered
    // exactly once per sweep before the swap, so the hazard cannot arise -- and
    // Jacobi iteration is order-independent, which is what makes the spec's
    // "assert bit-identical against the reference" gate possible later on.
    void swap_irradiance();

    // Copies b_irrad_prev -> b_irrad so receivers outside this frame's slice
    // keep their value across the swap. One GPU-side copy per sweep.
    void carry_irradiance();

    uint32_t count() const { return count_; }
    float    radius() const { return radius_; }
    float    spacing() const { return spacing_; }
    double   area() const { return area_; }
    uint32_t emissive_count() const { return emissive_count_; }
    uint32_t two_sided_count() const { return two_sided_count_; }
    const Bounds& bounds() const { return bounds_; }
    double   bake_seconds() const { return bake_seconds_; }
    std::size_t bytes() const;

    // The home triangle of each surfel. Not uploaded -- the cut derivation
    // (cuts.cpp) needs a surfel's own face to decide which side of an edge its
    // disc keeps, and there is nowhere else to recover it from once the bake has
    // flattened the set.
    const std::vector<uint32_t>&  tri_of() const { return tri_of_; }

    const std::vector<glm::vec4>& pos_rad() const { return pos_rad_; }
    const std::vector<uint32_t>&  normal() const { return normal_; }
    const std::vector<uint32_t>&  albedo() const { return albedo_; }
    const std::vector<uint32_t>&  emission() const { return emission_; }

    gl::Buffer& irradiance() { return b_irrad_; }
    gl::Buffer& irradiance_prev() { return b_irrad_prev_; }

    // Stalling read-back of the irradiance buffer. Diagnostics and gates only.
    std::vector<glm::vec4> read_irradiance() const;

private:
    void upload();

    uint32_t count_ = 0;
    float    radius_ = 0.0f;
    float    spacing_ = 0.0f;
    double   area_ = 0.0;
    uint32_t emissive_count_ = 0;
    uint32_t two_sided_count_ = 0;
    std::vector<uint32_t> tri_of_;
    double   bake_seconds_ = 0.0;
    Bounds   bounds_;

    std::vector<glm::vec4> pos_rad_;
    std::vector<uint32_t>  normal_;
    std::vector<uint32_t>  albedo_;
    std::vector<uint32_t>  emission_;

    gl::Buffer b_pos_rad_  {gl::BufferType::shader, gl::BufferUsage::static_draw};
    gl::Buffer b_normal_   {gl::BufferType::shader, gl::BufferUsage::static_draw};
    gl::Buffer b_albedo_   {gl::BufferType::shader, gl::BufferUsage::static_draw};
    gl::Buffer b_emission_ {gl::BufferType::shader, gl::BufferUsage::static_draw};
    gl::Buffer b_irrad_    {gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer b_irrad_prev_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
};

} // namespace sgi
