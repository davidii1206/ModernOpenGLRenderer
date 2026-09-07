#pragma once

// ---------------------------------------------------------------------------
// A uniform grid over the static surfel set.
//
// Built once on the CPU, because the set never moves in this example and a GPU
// counting sort would mean bringing back the PrefixSum that was deliberately
// dropped from this example's gpu_util. It moves to the GPU the moment surfels
// become dynamic; nothing about the layout below would change.
//
// FAT INSERTION is the load-bearing decision. A surfel goes into every cell its
// BOUNDING SPHERE touches, not just the cell holding its centre. That costs
// ~2-3x the entries and buys two things:
//
//   * the per-pixel gather can scan a small fixed window and be sure it has
//     seen every disc that reaches the query point, and
//   * a ray marching this grid can scan ONE cell per DDA step and still be
//     exact, because a disc that could intersect the cell is already listed in
//     it. That is the difference from example 38, which point-inserts and so
//     must scan a 3x3x3 window (or a 9-cell slab) at every step of every one of
//     its four LOD levels -- the cost its own comments blame for a latency-bound
//     kernel at ~10% warp occupancy.
//
// Fat insertion is only affordable because this example bakes ONE GLOBAL RADIUS
// (see surfels.hpp): a sphere of radius r spans at most two cells per axis when
// the cell is about a surfel spacing, so it lands in <= 8 cells and typically 2-3.
// With 38's adaptive per-surfel radii across four levels it would not be.
// ---------------------------------------------------------------------------

#include "surfels.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

namespace sgi {

// Binding points, on top of SurfelSet's 0..5 and the solver's 6..7.
enum GridBinding : uint32_t {
    kBindCellSC   = 8,   // uvec2[cells]    start, count
    kBindCellItem = 9,   // uint[entries]   surfel indices, cell-sorted
    kBindCellPR   = 10,  // vec4[entries]   cell-sorted pos.xyz + radius
    kBindMacro    = 21,  // uint[]          occupancy bit per 4x4x4 macro block
    kBindIrradFilt = 11, // vec4[N]         denoised irradiance, display only
    kBindLightGrad = 18, // vec4[N]         world-space gradient of light_vis, w = fit quality
};

class SurfelGrid {
public:
    // `cell_mul` is the cell size in surfel SPACINGS. The spacing is the natural
    // unit: the bake ties it to the radius exactly (spacing = r*sqrt(PI)), so one
    // number controls candidate count and window size together.
    void build(const SurfelSet& set, float cell_mul);

    void bind() const;

    bool         valid() const { return count_ > 0; }
    glm::ivec3   res() const { return res_; }
    glm::ivec3   macro_res() const { return macro_res_; }
    uint32_t     macro_occupied() const { return macro_occupied_; }
    glm::vec3    min() const { return min_; }
    float        cell() const { return cell_; }
    float        inv_cell() const { return 1.0f / cell_; }
    uint32_t     cells() const { return uint32_t(res_.x) * uint32_t(res_.y) * uint32_t(res_.z); }
    uint32_t     occupied() const { return occupied_; }
    uint32_t     entries() const { return entries_; }
    uint32_t     max_per_cell() const { return max_per_cell_; }
    float        mean_per_occupied() const {
        return occupied_ ? float(entries_) / float(occupied_) : 0.0f;
    }
    double       build_seconds() const { return build_seconds_; }
    std::size_t  bytes() const;

private:
    uint32_t   count_ = 0;
    glm::ivec3 res_{0};
    glm::vec3  min_{0.0f};
    float      cell_ = 0.0f;
    uint32_t   occupied_ = 0, entries_ = 0, max_per_cell_ = 0;
    // One bit per 4x4x4 block of cells. A frustum query over a wide area light
    // touches tens of thousands of cells, almost all of them in empty space --
    // Cornell's interior is hollow and the grid is 16% occupied. Testing a macro
    // block first turns that into a few hundred probes.
    glm::ivec3 macro_res_{0};
    uint32_t   macro_occupied_ = 0;
    double     build_seconds_ = 0.0;

    gl::Buffer b_sc_  {gl::BufferType::shader, gl::BufferUsage::static_draw};
    gl::Buffer b_item_{gl::BufferType::shader, gl::BufferUsage::static_draw};
    gl::Buffer b_pr_  {gl::BufferType::shader, gl::BufferUsage::static_draw};
    gl::Buffer b_macro_{gl::BufferType::shader, gl::BufferUsage::static_draw};
};

} // namespace sgi
