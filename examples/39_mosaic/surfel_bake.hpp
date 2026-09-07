#pragma once

// ---------------------------------------------------------------------------
// M1 — object-space surfel bake.
//
// Each (model, mesh) pair gets one surfel set, sampled in the mesh's OWN vertex
// space and quantized to the mesh AABB. Instances then transform the set into
// world space per frame, which is what makes a moving object's cached
// irradiance travel with it instead of being invalidated.
//
// Placement is variable-radius Poisson-disk dart throwing, following the
// approach in example 38 (main.cpp:549-716): pick a triangle weighted by
// area/h^2, place a uniform barycentric point, and accept only if no accepted
// same-surface sample of the same connected component is within the local
// spacing. Spacing is curvature-adaptive, so flat walls stay coarse and curved
// detail refines.
//
// Unlike example 37 (whose surfels carry glTF factor colours only, leaving
// every Sponza surfel white), albedo and emissive are SAMPLED FROM THE MESH
// TEXTURES at each surfel's interpolated UV.
// ---------------------------------------------------------------------------

#include "mosaic.hpp"

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>
#include <glm/glm.hpp>

#include <map>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace mosaic {

// Packed 16-byte object-space surfel record. Layout matches the decoder in
// shaders/common/surfel.glsl, and mirrors example 37's proven packing
// (validated there to ~4e-5 max position error) so the two interchange.
//
//   .x = pos_x_q16 << 16 | radius_q16
//   .y = pos_y_q16 << 16 | pos_z_q16
//   .z = albedo R8 G8 B8 | flags A8
//   .w = octahedral normal, snorm16 pair
struct PackedSurfel {
    uint32_t x, y, z, w;
};
static_assert(sizeof(PackedSurfel) == 16, "PackedSurfel must stay 16 bytes");

enum SurfelFlags : uint32_t {
    kSurfelEmissive = 1u << 0,
    kSurfelTwoSided = 1u << 1,   // foliage / thin geometry: gathers both sides
};

// Emissive is kept in TWO forms, because two different consumers want different
// access patterns:
//
//  - a DENSE RGB9E5 array parallel to `packed`, so S2's per-surfel direct
//    lighting can look up any surfel's emission with one 4-byte fetch;
//  - a SPARSE list, so M5's emitter-proxy extraction and S4's RIS candidate
//    generation can iterate only the surfaces that actually emit.
//
// Dense costs 4 bytes/surfel (2.6 MB at Sponza's 660 k), which is cheaper than
// the branchy indirection a sparse-only layout would force on every surfel.
struct EmissiveSurfel {
    uint32_t index;
    float r, g, b;
};
static_assert(sizeof(EmissiveSurfel) == 16, "EmissiveSurfel must stay 16 bytes");

// CPU-side decoders, mirroring shaders/common/surfel.glsl. Needed by M5's
// emitter-proxy extraction, which fits rectangles to the baked emissive surfels
// rather than rasterizing emissive textures into UV space as the spec does.
struct SurfelSet;
glm::vec3 surfel_position(const SurfelSet& set, const PackedSurfel& p);
glm::vec3 surfel_normal(const PackedSurfel& p);
float     surfel_radius(const SurfelSet& set, const PackedSurfel& p);

// Shared exponent RGB9E5 (matches GL_RGB9_E5 / the GLSL decoder in surfel.glsl).
uint32_t pack_rgb9e5(const glm::vec3& c);

// One surfel set: all four LODs of one mesh, concatenated.
struct SurfelSet {
    glm::vec3 aabb_min{0.0f};
    glm::vec3 aabb_max{0.0f};
    float radius_scale = 1.0f;      // quantization domain for the radius field

    struct Lod {
        uint32_t offset = 0;
        uint32_t count = 0;
        float spacing = 0.0f;       // target Poisson spacing of this level
        float mean_radius = 0.0f;
    };
    Lod lods[Config::kSurfelLods];

    std::vector<PackedSurfel> packed;   // LOD 0 (finest) first
    // Index of the covering surfel one LOD coarser, or 0xFFFFFFFF at the
    // coarsest level. LOD crossfade (spec §4) seeds a level from its parent's
    // radiance, so the link has to be baked, not searched at runtime.
    std::vector<uint32_t> parent;
    std::vector<uint32_t> emissive_dense;    // RGB9E5, parallel to `packed`
    std::vector<EmissiveSurfel> emissive;    // only the surfels that emit

    uint32_t total() const { return uint32_t(packed.size()); }
};

// Bake parameters.
//
// `world_spacing` is LOD 0's target Poisson spacing in WORLD units. Deriving it
// from a surfel budget rather than from the cascade size is load-bearing: a
// spacing chosen from the cascade cell alone oversamples Sponza by two orders of
// magnitude (every one of its 103 meshes saturates the per-mesh cap), because
// the cascade size says nothing about how much surface area the scene contains.
// Use spacing_for_budget() to pick it.
struct BakeParams {
    float world_spacing = 0.05f;
    float curvature_adapt = 8.0f;    // h = base / (1 + adapt * kappa * base)
    float curvature_floor = 0.25f;   // never refine below floor * base
    uint32_t max_surfels_per_lod = 200000;   // safety valve, not the usual limit
    uint32_t seed = 12345;
};

// Total world-space surface area of everything the instance list draws.
double scene_surface_area(const std::vector<Instance>& instances,
                          const std::vector<const gfx::Model*>& models);

// LOD-0 spacing that yields roughly `target` surfels over `area`. A Poisson-disk
// set at spacing h covers about h^2 of surface per sample.
inline float spacing_for_budget(double area, uint32_t target) {
    if (area <= 0.0 || target == 0) return 0.05f;
    return float(std::sqrt(area / double(target)));
}

// CPU-side decoded texture, read back from GL because gfx::Model discards the
// decoded bytes after upload. Values are linearized on read.
struct CpuTexture {
    int width = 0, height = 0;
    std::vector<glm::vec4> texels;    // linear RGBA

    bool valid() const { return width > 0 && height > 0; }
    glm::vec4 sample(glm::vec2 uv) const;   // bilinear, repeat wrap
};

// Reads back every texture of a model into CPU memory. Indexed the same way as
// gfx::Model::texture(i), so ModelMaterialInfo's indices apply directly.
std::vector<CpuTexture> read_back_textures(const gfx::Model& model);

// Bakes one mesh. `textures` may be empty, in which case material factors are
// used alone. `object_scale` is the uniform scale of the instance transform that
// places this mesh: the set is stored in object space, so the world-space
// spacing target has to be converted into the mesh's own units.
SurfelSet bake_mesh(const gfx::Model& model, size_t mesh_index,
                    const std::vector<CpuTexture>& textures,
                    const BakeParams& params, float object_scale);

// --- Library ----------------------------------------------------------------

// All surfel sets referenced by the instance list, concatenated into one GPU
// buffer. A set is keyed by (model, mesh), so N instances of the same mesh share
// a single set — the point of storing surfels in object space.
class SurfelLibrary {
public:
    struct Entry {
        int model = 0;
        int mesh = 0;
        uint32_t base = 0;          // offset of this set inside the shared buffers
        float object_scale = 1.0f;  // uniform scale of the placing transform
        SurfelSet set;
    };

    // Bakes (or loads from cache) every distinct (model, mesh) the instances
    // reference, then uploads. `model_paths` is parallel to `models`.
    void build(const std::vector<Instance>& instances,
               const std::vector<const gfx::Model*>& models,
               const std::vector<std::string>& model_paths,
               const BakeParams& params);

    // Index into entries() for an instance, or -1 when the mesh had no geometry.
    int entry_for(const Instance& inst) const;

    const std::vector<Entry>& entries() const { return entries_; }
    gl::Buffer& packed_buffer() { return packed_; }
    gl::Buffer& parent_buffer() { return parent_; }
    gl::Buffer& emissive_buffer() { return emissive_; }
    gl::Buffer& emissive_dense_buffer() { return emissive_dense_; }

    uint32_t total_surfels() const { return total_surfels_; }
    uint32_t total_emissive() const { return total_emissive_; }
    double bake_seconds() const { return bake_seconds_; }
    uint32_t cached_sets() const { return cached_sets_; }

private:
    std::vector<Entry> entries_;
    std::map<std::pair<int, int>, int> lookup_;
    gl::Buffer packed_{gl::BufferType::shader, gl::BufferUsage::static_draw};
    gl::Buffer parent_{gl::BufferType::shader, gl::BufferUsage::static_draw};
    gl::Buffer emissive_{gl::BufferType::shader, gl::BufferUsage::static_draw};
    gl::Buffer emissive_dense_{gl::BufferType::shader, gl::BufferUsage::static_draw};
    uint32_t total_surfels_ = 0;
    uint32_t total_emissive_ = 0;
    uint32_t cached_sets_ = 0;
    double bake_seconds_ = 0.0;
};

// --- Disk cache -------------------------------------------------------------
//
// Invalidated on format version, source-file stamp (mtime ^ size), material
// hash, mesh index, and the bake parameters, following the scheme in example 37
// (main.cpp:600-709). Positions are baked in object space and therefore survive
// any instance transform change, so the instance list is deliberately NOT part
// of the key.
std::string cache_path(const std::string& model_path, size_t mesh_index);
bool save_cache(const std::string& path, const SurfelSet& set,
                const std::string& model_path, const gfx::Model& model,
                size_t mesh_index, const BakeParams& params, float object_scale);
bool load_cache(const std::string& path, SurfelSet& out,
                const std::string& model_path, const gfx::Model& model,
                size_t mesh_index, const BakeParams& params, float object_scale);

} // namespace mosaic
