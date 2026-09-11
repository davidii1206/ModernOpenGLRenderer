#pragma once

// ---------------------------------------------------------------------------
// M1 + M2 -- the brute-force solver, spec section 10 steps 1 and 2.
//
//   pass 0  bf_lout.comp      L_out = L_e + albedo * E_prev / PI, once per sweep
//   M1      bf_radiance.comp  all-pairs radiance, no occlusion
//   M2      bf_micro.comp     all-pairs candidates through a per-receiver
//                             hemi-octahedral microbuffer
//
// Both solvers are O(N^2) per sweep. The driver splits a sweep across frames so
// the app stays interactive, and ping-pongs the irradiance buffers at SWEEP
// boundaries, which makes each sweep one Jacobi iteration of the Neumann
// series: sweep 1 is direct light exactly, sweep k adds bounce k-1.
//
// Jacobi rather than in-place Gauss-Seidel for two reasons beyond section 7's.
// A GPU Gauss-Seidel is not merely order-dependent, it is NONDETERMINISTIC --
// the order is workgroup scheduling order -- which would make step 4's "assert
// bit-identical against the reference" impossible. And Jacobi's iterate k is
// exactly k bounces, so sweep 1 is precisely the quantity the direct-only path
// trace shows; a Gauss-Seidel iterate is somewhere between k and infinity
// depending on scheduling and matches nothing.
// ---------------------------------------------------------------------------

#include "gpu_util.hpp"
#include "surfels.hpp"
#include "grid.hpp"
#include "direct.hpp"
#include "cuts.hpp"

namespace sgi {

enum class Method : int {
    Radiance = 0,   // M1, no occlusion
    Micro    = 1,   // M2, microbuffer occlusion
};

struct SolveConfig {
    Method   method = Method::Micro;
    uint32_t budget = 2048;        // receivers per frame
    uint32_t max_sweeps = 8;       // 0 = run forever. Sweep 1 == direct light only.
    uint32_t ms = 16;              // microbuffer edge: 8 -> 64 buckets, 16 -> 256

    // Shared by BOTH solvers. Gate 5 compares them and needs these identical.
    float horizon = 0.02f;         // cos(theta) floor on the receiver side
    float plane_bias = 1.0f;       // in receiver radii
    float soft_eps = 1.0f;         // disc softening d^2 -> d^2 + eps*r^2
    // Radius beyond which a surfel lights but does not occlude, in world units.
    // 0 = unlimited. Simulates the FMM's U-list horizon without any FMM code.
    float near_radius = 0.0f;
    // March the grid's macro occupancy bitmask per bucket to find how far the
    // far field can reach before something is in the way. Only meaningful with
    // near_radius set, since without a horizon nothing is far.
    bool far_occlusion = true;

    // Force EVERY surfel to emit from both faces. Diagnostic only; off.
    //
    // This used to default to true, on the reasoning that Cycles' emission BSDF
    // is two-sided. It is -- but the switch is global, so it also made every
    // opaque surface emit its own outgoing radiance backwards: light through the
    // walls, and the boxes' lids lighting the sealed volume underneath them. A
    // material's own double-sided flag has been baked into the surfel record
    // since M0b and is now what the shaders read; this only forces it on.
    bool  two_sided = false;
    float emissive_scale = 1.0f;
    // The environment a bucket sees when it sees no geometry. `sky` is the
    // zenith radiance and `sky_ground` the radiance below the horizon, blended
    // over the bucket's world direction; the sun is a bright disc added on top.
    //
    // The sun rides in the environment rather than in the NEE path because that
    // is where its SHADOW comes from for free: a bucket that hits something
    // never asks what the sky looks like. The cost is that its shadow is only as
    // sharp as the microbuffer's 16x16, which is why u_sun_cos defaults to a
    // disc several degrees wide rather than the sun's real 0.53 -- an unresolved
    // source in 256 buckets aliases into a bucket-shaped shadow.
    glm::vec3 sky{0.0f};           // zenith radiance of an uncovered bucket
    glm::vec3 sky_ground{0.0f};    // radiance below the horizon
    glm::vec3 sun{0.0f};           // sun radiance, added inside its disc
    glm::vec3 sun_dir{0.0f, 1.0f, 0.0f};   // world direction TO the sun
    float     sun_cos = 0.999f;    // cos of the sun's angular radius
    // Who owns the sun. NEE gives it a sharp shadow through the cone bitmask;
    // the microbuffer gives it a 16x16-bucket one for free. Exactly one of them
    // must have it, or it is counted twice.
    bool      sun_nee = true;
    // True when NEE has a sun to trace, whatever the emitter buffer holds.
    bool sun_is_nee_light() const {
        return sun_nee && (sun.r > 0.0f || sun.g > 0.0f || sun.b > 0.0f);
    }
    float     sun_dist = 0.0f;     // where NEE places its stand-in rectangle
    float     sun_half = 0.0f;     // and that rectangle's half-extent

    float depth_tol_radii = 2.0f;  // distance from the winner's plane, in radii
    float normal_tol = 0.7f;       // cos of the largest normal disagreement kept
    bool  no_occlusion = false;    // gate 5
    int   rotate = 1;              // 0 none, 1 static per surfel, 2 per surfel per frame

    bool  debug_constant = false;  // gate 3
    float debug_radiance = 1.0f;

    // Split the direct term out of the microbuffer and sample the proxied
    // emitters explicitly (nee_direct.comp). A hard partition: their emission is
    // removed from lout, their coverage is not, so they still occlude.
    bool  nee = true;
    // Take the IMAGE's direct term per pixel instead of from the cache. The
    // cache still carries it for the bounce transport either way, so this changes
    // only what the reconstruction reads -- and it lifts the direct term off the
    // surfel Nyquist limit, which is what makes a shadow silhouette clean.
    bool  nee_pixel = false;
    // Surfel slab half-thickness, in radii. An edge-on disc projects to a line
    // and occludes nothing, so grazing surfaces leak; modelling a surfel as a
    // thin oblate spheroid gives a silhouette semi-minor axis of
    // r*(|cos| + thick*|sin|), which is exact for that solid. Replaces the flat
    // minor-axis clamp, whose constant over-occlusion at grazing incidence was
    // the ragged dark band along every concave edge.
    float nee_thick = 0.25f;
    // A flat surface does not shadow itself. An occluder whose normal agrees
    // with the receiver's to better than `self_cos` AND whose centre is within
    // `self_tol` radii of the receiver's tangent plane is the receiver's own
    // surface and is skipped. This is what lets the bias be small.
    float nee_self_cos = 0.9f;
    float nee_self_tol = 1.0f;
    // Occluder radius scale for the visibility query only -- see nee.glsl. The
    // bake's radius is the ENERGY radius (sum(pi r^2) == A); circles at that
    // radius do not tile, so they do not seal. Coverage goes as the square.
    float nee_occ = 2.0f;
    // Clip occluder discs at the source mesh's sharp and boundary edges
    // (cuts.cpp). Without it a silhouette is dilated by one surfel radius and
    // scalloped at the surfel spacing; with it the boundary is exact and u_occ
    // is free to inflate the interior as much as sealing requires.
    bool  nee_cuts = true;
    float nee_bias = 0.05f;       // receiver offset along its normal, in radii
    // Per-pixel direct only: skip the grid march where every surfel that feeds
    // the pixel agrees the light is fully visible or fully blocked, and take the
    // rectangle's closed form instead. The value is the agreement margin; 0 is
    // off. See nee_pixel.comp.
    float nee_skip = 0.0f;

    bool  running = true;
};

// What a completed sweep looked like. Filled from a stalling read-back, so it is
// gated behind a toggle and only sampled at sweep boundaries.
struct SolveStats {
    double mean = 0.0;             // mean irradiance luminance
    double peak = 0.0;
    double flux = 0.0;             // sum a_i * E_i, in watts
    double delta = 0.0;            // mean |E_k - E_{k-1}|
    double rel_delta = 0.0;
    uint32_t nonzero = 0;
    bool valid = false;
};

// Exact per-bucket table for an MS x MS hemi-octahedral microbuffer.
//
// Two vec4 per bucket: (dir.xyz in the tangent frame, dw) and (wcos, 0, 0, 0),
// where dw is the texel's solid angle by Girard's theorem and wcos is
// INTEGRAL(cos(theta) dOmega) over the texel. See shaders/common/micro.glsl for
// why neither can be assumed uniform.
std::vector<glm::vec4> build_bucket_table(uint32_t ms);

// Binding points used by the solver on top of SurfelSet's 0..5.
enum SolverBinding : uint32_t {
    kBindLout   = 6,
    kBindBucket = 7,
    kBindLightVis = 16, // float[N]     cosine-weighted emitter visibility
    kBindDirect   = 20, // vec4[N]      NEE direct irradiance
    kBindBlkRad   = 24, // uint[blocks*4]  order-0 multipole: sum(L_out), count
    kBindLightMax = 17, // uint[1]      running max of the above, bit-cast float
};

class Solver {
public:
    bool init();
    bool poll();

    // The direct pass needs the grid to find occluders and the emitter proxies to
    // sample. Both are static, so they are attached once rather than threaded
    // through every call.
    void attach(const SurfelGrid* grid, const EmitterSet* emitters,
                const CutSet* cuts = nullptr) {
        grid_ = grid; emitters_ = emitters; cuts_ = cuts; direct_dirty_ = true;
    }

    void reset(SurfelSet& set);

    // One frame's worth of work: a slice of at most cfg.budget receivers.
    void step(SurfelSet& set, const SolveConfig& cfg, uint32_t frame, bool collect_stats);

    // Runs `n` complete sweeps back to back, ignoring the budget.
    void run_sweeps(SurfelSet& set, const SolveConfig& cfg, uint32_t n, uint32_t frame);

    uint32_t sweeps() const { return sweeps_; }
    uint32_t cursor() const { return cursor_; }
    bool converged(const SolveConfig& c) const {
        return c.max_sweeps != 0 && sweeps_ >= c.max_sweeps;
    }
    const SolveStats& stats() const { return stats_; }
    PassTimer& timer() { return *timer_; }

    // Diagnostics for the bucket table, printed at startup and by gate 3.
    void bucket_sums(uint32_t ms, double& sum_dw, double& sum_wcos);

    // Emitter visibility per surfel; valid once a Micro sweep has run.
    void bind_light_vis() const {
        b_light_.bind_base(kBindLightVis);
        b_light_max_.bind_base(kBindLightMax);
    }
    bool light_vis_valid() const { return light_count_ != 0; }
    // The occluder query the per-pixel direct pass needs: the same grid and
    // proxies this solver was attached to. Handed out rather than re-plumbed so
    // the two passes cannot end up querying different structures.
    const SurfelGrid* grid() const { return grid_; }
    const EmitterSet* emitters() const { return emitters_; }

private:
    void ensure_buffers(const SurfelSet& set, uint32_t ms);
    void run_lout(SurfelSet& set, const SolveConfig& cfg);
    void run_direct(SurfelSet& set, const SolveConfig& cfg);
    bool nee_active(const SolveConfig& cfg) const;
    void dispatch(SurfelSet& set, const SolveConfig& cfg,
                  uint32_t first, uint32_t slice, uint32_t frame);
    void end_sweep(SurfelSet& set, bool collect_stats);

    Pipeline lout_prog_, radiance_, micro_, direct_, blk_prog_;
    std::unique_ptr<PassTimer> timer_;

    gl::Buffer b_lout_  {gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer b_bucket_{gl::BufferType::shader, gl::BufferUsage::static_draw};
    gl::Buffer b_light_ {gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer b_light_max_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    gl::Buffer b_direct_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    // Order-0 multipole, one (sum L_out, count) per macro block.
    gl::Buffer b_blk_   {gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    uint32_t   blk_words_ = 0;
    uint32_t   direct_count_ = 0;
    // The direct term depends only on geometry and emission, not on the sweep's
    // irradiance, so it is computed once per solve rather than once per sweep.
    bool       direct_dirty_ = true;
    float      direct_thick_ = -1.0f, direct_bias_ = -1.0f, direct_scale_ = -1.0f;
    float      direct_self_cos_ = -2.0f, direct_self_tol_ = -1.0f;
    float      direct_occ_ = -1.0f;
    int        direct_cuts_ = -1;
    const SurfelGrid* grid_ = nullptr;
    const EmitterSet* emitters_ = nullptr;
    const CutSet*     cuts_ = nullptr;
    uint32_t   light_count_ = 0;
    uint32_t   lout_count_ = 0;
    uint32_t   bucket_ms_ = 0;

    uint32_t cursor_ = 0;
    uint32_t sweeps_ = 0;
    SolveStats stats_;
    std::vector<glm::vec4> snapshot_;
};


} // namespace sgi
