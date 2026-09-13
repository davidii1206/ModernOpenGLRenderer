#include "fmm.hpp"

#include <gl/gl.hpp>
#include <gllib/log.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>

namespace sgi {

namespace {

inline uint32_t lin(const glm::ivec3& c, const glm::ivec3& res) {
    return uint32_t(c.x) + uint32_t(res.x) * (uint32_t(c.y) + uint32_t(res.y) * uint32_t(c.z));
}

// The V-list test of spec section 3.3, run once on the CPU so the count is an
// assertion rather than a hope. A target at parent-relative offset b sees the
// 6x6x6 box of its parent's neighbourhood; the 27 cells adjacent to it are its
// own U-list and belong to the microbuffer.
int vlist_count(const glm::ivec3& b) {
    int n = 0;
    for (int dz = -2 - b.z; dz <= 3 - b.z; ++dz)
        for (int dy = -2 - b.y; dy <= 3 - b.y; ++dy)
            for (int dx = -2 - b.x; dx <= 3 - b.x; ++dx) {
                const int m = std::max(std::max(std::abs(dx), std::abs(dy)), std::abs(dz));
                if (m > 1) ++n;
            }
    return n;
}

} // namespace

void FmmTree::build(const SurfelSet& set, const SurfelGrid& grid) {
    const auto t0 = std::chrono::steady_clock::now();
    levels_.clear();
    total_cells_ = total_slots_ = 0;
    if (!grid.valid() || set.count() == 0) return;

    // The eight parent-relative positions all have to produce 189, or the
    // stencil is not the one the O(N) argument assumes.
    for (int z = 0; z < 2; ++z)
        for (int y = 0; y < 2; ++y)
            for (int x = 0; x < 2; ++x)
                assert(vlist_count(glm::ivec3(x, y, z)) == kVListSize);

    // Level 0 occupancy: the OWNER cell of each surfel, which is the cell
    // containing its centre. Fat insertion puts a surfel in about ten cells and
    // exactly one of them owns it, so this is a partition and P2M cannot double
    // count -- see grid.cpp.
    std::vector<glm::ivec3> res_of;
    glm::ivec3 r = grid.res();
    res_of.push_back(r);
    while (r.x > 1 || r.y > 1 || r.z > 1) {
        r = glm::max(glm::ivec3(1), (r + 1) / 2);
        res_of.push_back(r);
    }

    std::vector<std::vector<uint8_t>> occ(res_of.size());
    for (std::size_t l = 0; l < res_of.size(); ++l)
        occ[l].assign(std::size_t(res_of[l].x) * res_of[l].y * res_of[l].z, 0);

    const glm::vec3 mn = grid.min();
    const float inv = grid.inv_cell();
    for (uint32_t i = 0; i < set.count(); ++i) {
        const glm::ivec3 c = glm::clamp(
            glm::ivec3(glm::floor((glm::vec3(set.pos_rad()[i]) - mn) * inv)),
            glm::ivec3(0), res_of[0] - 1);
        occ[0][lin(c, res_of[0])] = 1;
    }
    // A parent is occupied when any child is. Nothing else can make it so: the
    // multipole is a sum over its children and an empty one contributes zero.
    for (std::size_t l = 1; l < res_of.size(); ++l) {
        const glm::ivec3 pr = res_of[l], cr = res_of[l - 1];
        for (int z = 0; z < cr.z; ++z)
            for (int y = 0; y < cr.y; ++y)
                for (int x = 0; x < cr.x; ++x)
                    if (occ[l - 1][lin(glm::ivec3(x, y, z), cr)])
                        occ[l][lin(glm::ivec3(x / 2, y / 2, z / 2), pr)] = 1;
    }

    // Compact. The slot map is dense and the coefficients are not.
    std::vector<int32_t>  slot;
    std::vector<uint32_t> cell;
    for (std::size_t l = 0; l < res_of.size(); ++l) {
        Level lv;
        lv.res = res_of[l];
        lv.cells = uint32_t(occ[l].size());
        lv.slot_base = uint32_t(slot.size());
        lv.coeff_base = uint32_t(cell.size());
        slot.resize(slot.size() + lv.cells, -1);
        for (uint32_t c = 0; c < lv.cells; ++c) {
            if (!occ[l][c]) continue;
            slot[lv.slot_base + c] = int32_t(cell.size() - lv.coeff_base);
            cell.push_back(c);
        }
        lv.slots = uint32_t(cell.size()) - lv.coeff_base;
        levels_.push_back(lv);
    }
    total_cells_ = uint32_t(slot.size());
    total_slots_ = uint32_t(cell.size());

    // One buffer: the slot map, the slot->cell map, then the two coefficient
    // regions zeroed. The coefficients are floats read through uintBitsToFloat,
    // so the whole thing is one word array.
    std::vector<uint32_t> tree(std::size_t(lsh_off()) + std::size_t(total_slots_) * 27u, 0u);
    std::memcpy(tree.data(), slot.data(), slot.size() * sizeof(int32_t));
    std::memcpy(tree.data() + cell_off(), cell.data(), cell.size() * sizeof(uint32_t));
    b_tree_.data(tree.data(), tree.size() * sizeof(uint32_t));

    build_seconds_ = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0).count();

    gllib::logf(gllib::LogLevel::info,
                "fmm: %zu levels, %u cells -> %u occupied (%.1f%%), %.1f KB, %.3f s",
                levels_.size(), total_cells_, total_slots_,
                100.0 * double(total_slots_) / double(std::max(1u, total_cells_)),
                double(bytes()) / 1024.0, build_seconds_);
    for (std::size_t l = 0; l < levels_.size(); ++l)
        gllib::logf(gllib::LogLevel::info, "  level %zu: %dx%dx%d, %u occupied",
                    l, levels_[l].res.x, levels_[l].res.y, levels_[l].res.z,
                    levels_[l].slots);
}

void FmmTree::bind() const { b_tree_.bind_base(kBindFmmTree); }

std::size_t FmmTree::bytes() const {
    return (std::size_t(lsh_off()) + std::size_t(total_slots_) * 27u) * sizeof(uint32_t);
}

} // namespace sgi
