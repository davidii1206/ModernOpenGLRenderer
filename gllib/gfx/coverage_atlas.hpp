#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace gfx {

class Model;

// Configuration for the mesh -> coverage-atlas pipeline.
//
// All values are user-selectable per model. `texel_density` is the primary
// resolution control (texels per world unit); a large building at the same
// density automatically receives more texels than a small model, and you can
// tune it per asset. Everything used to be fixed at compile time.
struct CoverageAtlasConfig {
    float texel_density   = 256.0f;  // texels per unit (base atlas resolution; <=0 = auto)
    int   auto_target     = 1024;    // auto mode: texels across the model's longest axis
    float budget_texels   = 64.0e6f; // global atlas texel budget (caps density)
    int   min_tex         = 4;       // min patch texture size (texels)
    int   max_tex         = 1024;    // max patch texture size (texels)
    float mip_tol_frac    = 0.004f;  // mip subdivide tolerance (fraction of global range)
    int   mip_leaf_tile   = 1;       // finest mip node size (texels); 1 = per-texel detail
    int   min_patch_size  = 200;     // merge pass target: absorb patches below this
    float epsilon         = 0.001f;  // occlusion test depth tolerance
    float axis_threshold  = 0.5f;    // min |dot(normal, axis)| for merge candidates
    float normal_crease_angle = 30.0f;  // hard-edge threshold (degrees); edges
                                        // softer than this are smoothed, 0 = flat
    bool  rotate_model_x  = true;    // rotate scene 90° about X (glTF Y-up -> upright)
};

// CPU-only mesh -> occlusion-free patch atlas pipeline. Takes a loaded
// gfx::Model (LOD0 meshes only), welds and clusters it into occlusion-free
// convex-ish patches, packs them into a texture atlas, builds a BVH, rasterises
// depth/thickness/UV coverage and emits the three adaptive MIP4 sparse
// mip-chains. No OpenGL is required (the model must already be loaded).
//
// Result invariants (see docs/mip4.md):
//   - Level-0 chain node index == patch index in patches() (pack order)
//   - Patches subdivide 1->4 while partially covered or above value tolerance
//   - Leaf coverage granularity is `mip_leaf_tile` texels (default 1 = per-texel)
class CoverageAtlas {
public:
    struct Vec3Tri {
        uint32_t v[3];
    };

    struct Patch {
        int   id   = 0;    // == position in patches() (pack order)
        int   axis = 0;    // dominant projection axis (0..5, +X..-Z)
        glm::vec3 aabb_min{0.0f}, aabb_max{0.0f};
        glm::vec3 basis_u{1, 0, 0}, basis_v{0, 1, 0}, basis_w{0, 0, 1};
        glm::vec2 proj_min{0.0f};     // min of the 2D projection bounds
        glm::vec2 proj_size{0.0f};    // (proj_max - proj_min)
        int tex_w = 0, tex_h = 0;      // patch texture size (texels)
        int atlas_x = 0, atlas_y = 0;  // atlas placement
        bool double_sided = false;     // from the source submesh's material (glTF doubleSided)
        std::vector<int> tris;         // indices into triangles()
    };

    struct BVHNode {
        glm::vec3 aabb_min{0.0f}, aabb_max{0.0f};
        uint16_t q_min[3] = {0, 0, 0}, q_max[3] = {0, 0, 0};
        union {
            uint32_t patch_index;    // leaf
            uint32_t right_offset;   // internal (relative)
        };
        uint32_t is_leaf = 0;
    };

    struct MipLevel {
        uint32_t w = 0, h = 0;
        std::vector<uint8_t>  data;   // w*h*channels value texel
        std::vector<uint32_t> meta;   // w*h traversal-metadata texel
    };

    struct MipChain {
        int    channels = 0;
        int    leaf_tile = 1;   // finest node size (texels); matches config.mip_leaf_tile
        int    bytes_per_channel = 1;  // 1 = 8-bit value texel, 2 = 16-bit (depth)
        float  qmin[2] = {0, 0}, qmax[2] = {0, 0};  // per-channel quant range
        std::vector<MipLevel> levels;
    };

    explicit CoverageAtlas(const CoverageAtlasConfig& config = {});
    ~CoverageAtlas() = default;
    CoverageAtlas(const CoverageAtlas&) = delete;
    CoverageAtlas& operator=(const CoverageAtlas&) = delete;

    // Run the pipeline. Returns false if the model has no geometry.
    bool build(const gfx::Model& model);

    // Replace the pipeline config (e.g. texel_density for a rebuild).
    void set_config(const CoverageAtlasConfig& config) { config_ = config; }

    // Write patch_table.bin, patch_tris.bin, bvh_nodes.bin, atlas_depth.bin,
    // atlas_thickness.bin, atlas_uv.bin, atlas_state.bin and patch_summary.txt
    // into dir (created if missing). Returns false on file I/O failure.
    bool write_files(const std::string& dir = ".") const;

    // Restore a previously written snapshot (atlas_state.bin) from dir without
    // re-running the pipeline. Returns false if the snapshot is missing or from
    // an incompatible version (caller should then build() + write_files()).
    bool load_files(const std::string& dir);

    const CoverageAtlasConfig& config() const { return config_; }
    const std::vector<glm::vec3>& positions() const { return positions_; }
    const std::vector<glm::vec3>& normals() const { return normals_; }
    const std::vector<glm::vec2>& uvs() const { return uvs_; }
    const std::vector<Vec3Tri>&   triangles() const { return triangles_; }
    const std::vector<Patch>&     patches() const { return patches_; }
    const std::vector<BVHNode>&   bvh_nodes() const { return bvh_nodes_; }
    int  atlas_width() const { return atlas_w_; }
    int  atlas_height() const { return atlas_h_; }
    float final_density() const { return final_density_; }

    const MipChain& depth_chain() const { return depth_chain_; }
    const MipChain& thickness_chain() const { return thickness_chain_; }
    const MipChain& uv_chain() const { return uv_chain_; }
    const MipChain& normal_chain() const { return normal_chain_; }

private:
    CoverageAtlasConfig config_;
    std::vector<glm::vec3> positions_, normals_;
    std::vector<glm::vec2> uvs_;
    std::vector<Vec3Tri>   triangles_;
    std::vector<Patch>     patches_;
    std::vector<BVHNode>   bvh_nodes_;
    int    atlas_w_ = 0, atlas_h_ = 0;
    float  final_density_ = 0.0f;
    MipChain depth_chain_, thickness_chain_, uv_chain_, normal_chain_;
};

} // namespace gfx
