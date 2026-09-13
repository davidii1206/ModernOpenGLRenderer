#pragma once

#include "grid.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

namespace sgi {

// The FMM's own binding points. The tree's four arrays are bound whole and the
// passes index into them with a per-level base, so a level change is a uniform
// and not a rebind.
// ONE binding, four regions. bf_micro.comp sits at this hardware's 16-block
// limit for a compute stage, so the tree gets one slot and offsets do the rest.
enum FmmBinding : uint32_t {
    kBindFmmTree = 12,  // uint[]: slot map, cell map, Ilm, Lsh
};

// The level-uniform octree of spec section 3.3, built over the surfel grid.
//
// Level 0 is the grid's own cell, which is the leaf: its 27-cell U-list is the
// near field the microbuffer already resolves, so the V-list at every level is
// exactly what the microbuffer does NOT see. Each level up halves the resolution
// in each axis until the root is a single cell.
//
// Only OCCUPIED cells are stored. A dense array at level 0 would be 1.55M cells
// on Sponza against 253k occupied, and the coefficients are 108 bytes each, so
// dense costs 335 MB to store mostly zeros. The slot map is the dense part -- one
// int per cell, 6 MB -- and everything expensive hangs off it compactly.
//
// The structure is built on the CPU because the grid is. Dynamic geometry moves
// both to the GPU together; splitting them now would mean two builds to keep in
// agreement and no way to assert one against the other.
class FmmTree {
public:
    struct Level {
        glm::ivec3 res{0};
        uint32_t   cells = 0;
        uint32_t   slots = 0;       // occupied cells
        uint32_t   slot_base = 0;   // offset into the slot map, in cells
        uint32_t   coeff_base = 0;  // offset into Ilm/Lsh, in CELLS (x27 for floats)
    };

    void build(const SurfelSet& set, const SurfelGrid& grid);
    void bind() const;

    bool valid() const { return !levels_.empty(); }
    uint32_t levels() const { return uint32_t(levels_.size()); }
    const Level& level(uint32_t l) const { return levels_[l]; }
    uint32_t total_slots() const { return total_slots_; }
    // Word offsets of the three regions after the slot map.
    uint32_t cell_off() const { return total_cells_; }
    uint32_t ilm_off() const { return total_cells_ + total_slots_; }
    uint32_t lsh_off() const { return ilm_off() + total_slots_ * 27u; }
    double build_seconds() const { return build_seconds_; }
    std::size_t bytes() const;

private:
    std::vector<Level> levels_;
    uint32_t total_cells_ = 0, total_slots_ = 0;
    double   build_seconds_ = 0.0;

    gl::Buffer b_tree_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
};

// 6^3 - 3^3. Asserted at build time, because a V-list that is quietly 188 or 190
// long is a brightness error of half a percent on a grid pattern and nothing else
// -- spec section 10 names it as the failure mode worth an assertion.
constexpr int kVListSize = 189;

} // namespace sgi
