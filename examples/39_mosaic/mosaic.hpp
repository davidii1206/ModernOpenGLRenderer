#pragma once

// ---------------------------------------------------------------------------
// MOSAIC — Micro-rendered Object-Space Amortized Irradiance Cache
//
// Shared scene/instance types and pipeline configuration. See implementation.md
// for the stage table and "Mosaic Lighting.md" (repo root) for the design spec.
// ---------------------------------------------------------------------------

#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace mosaic {

// --- Instance flags --------------------------------------------------------
enum InstanceFlags : uint32_t {
    kInstanceStatic  = 0u,
    kInstanceDynamic = 1u << 0,   // transform changes per frame (rigid mover)
    kInstanceSkinned = 1u << 1,   // reserved for M6; surfels skinned on GPU
    kInstanceVolatile= 1u << 2,   // deforming, needs periodic surfel resample
};

// One drawable: a mesh of a model placed in the world.
//
// The transform is kept separate from the surfel set on purpose: surfels are
// stored in OBJECT space and transformed per frame, so moving an instance never
// invalidates its cached irradiance (the load-bearing property of the design).
struct Instance {
    int       model = 0;          // index into the app's model list
    int       mesh = 0;           // mesh index within that model
    int       material = -1;      // material index within that model
    uint32_t  flags = kInstanceStatic;

    glm::mat4 xform{1.0f};
    glm::mat4 prev_xform{1.0f};

    // Animation for the rigid movers; ignored when kInstanceDynamic is unset.
    glm::vec3 orbit_center{0.0f};
    glm::vec3 orbit_axis{0.0f, 1.0f, 0.0f};
    float     orbit_radius = 0.0f;
    float     orbit_speed = 0.0f;
    float     orbit_phase = 0.0f;
    float     spin_speed = 0.0f;

    glm::mat3 normal_matrix() const {
        return glm::inverseTranspose(glm::mat3(xform));
    }
};

// Axis-aligned bounds helper used for scene fitting and instance culling.
struct Bounds {
    glm::vec3 mn{ 1e30f};
    glm::vec3 mx{-1e30f};

    void add(const glm::vec3& p) { mn = glm::min(mn, p); mx = glm::max(mx, p); }
    bool valid() const { return mn.x <= mx.x; }
    glm::vec3 center() const { return 0.5f * (mn + mx); }
    glm::vec3 extent() const { return mx - mn; }
    float radius() const { return valid() ? 0.5f * glm::length(mx - mn) : 1.0f; }
};

// --- Pipeline configuration -------------------------------------------------
//
// Cell sizes are expressed in SCENE UNITS, derived at load time from the scene
// extent rather than hardcoded to the spec's 0.5/2/8 m: the Cornell box is
// normalized to a 2-unit cube while Sponza spans thousands of units, and a
// fixed metric grid is meaningless for both.
struct Config {
    // S1 cluster clipmap: 3 camera-centred cascades.
    //
    // 64^3 = 262144 cells is EXACTLY the limit of the single-workgroup
    // block-sums scan (1024 blocks x 256). Raising kClipRes requires a
    // multi-level scan first — see gpu_util.cpp.
    static constexpr int kCascades = 3;
    static constexpr int kClipRes  = 64;
    static constexpr int kClipCells = kClipRes * kClipRes * kClipRes;
    static constexpr float kCascadeRatio = 4.0f;   // cell size multiplier per level

    // S1 occupancy clipmap (visibility only, carries no radiance).
    static constexpr int kOccRes = 128;

    // Surfel LOD spacing ratios, relative to the finest level.
    // Spec §2A: 4 cm / 8 cm / 25 cm / 1 m.
    static constexpr int kSurfelLods = 4;
    static constexpr float kLodSpacing[kSurfelLods] = {1.0f, 2.0f, 6.25f, 25.0f};

    // S3 scheduler.
    static constexpr uint32_t kMaxLiveSurfels = 256u * 1024u;
    uint32_t update_budget = 8192;

    // S5 micro-buffer edge length (16 -> 12 -> 8 are the spec's scaling knobs).
    uint32_t micro_size = 16;

    float  cascade0_cell = 0.0f;   // filled in at scene load
    float  surfel_spacing = 0.0f;  // finest LOD target spacing

    // How many cascades this scene actually needs.
    //
    // Cascades exist to cover distance cheaply, so building all three for a
    // scene that fits inside the first one is pure waste — and actively harmful:
    // the outer cascade's cells are then so large that the whole scene lands in
    // ONE cell, and a cluster pass that gives each cell a single thread spends
    // tens of milliseconds walking it. Sized from the scene extent at load.
    int cascades = kCascades;

    float cascade_cell(int l) const {
        float c = cascade0_cell;
        for (int i = 0; i < l; ++i) c *= kCascadeRatio;
        return c;
    }
    float cascade_coverage(int l) const { return cascade_cell(l) * float(kClipRes); }
};

} // namespace mosaic
