#include <cassert>
#include "grid.hpp"

#include <gl/gl.hpp>
#include <gllib/log.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace sgi {

void SurfelGrid::build(const SurfelSet& set, float cell_mul) {
    const auto t0 = std::chrono::steady_clock::now();

    count_ = set.count();
    if (count_ == 0) return;

    const float r = set.radius();
    cell_ = std::max(set.spacing() * std::max(cell_mul, 0.05f), 1e-5f);

    // Pad by one radius so a surfel on the bounding box's face still has every
    // cell its sphere touches inside the grid, and by one cell so the gather's
    // window never has to clamp against a surface that sits exactly on the edge.
    const Bounds& b = set.bounds();
    const float pad = r + cell_;
    min_ = b.mn - glm::vec3(pad);
    const glm::vec3 ext = (b.mx + glm::vec3(pad)) - min_;
    res_ = glm::max(glm::ivec3(glm::ceil(ext / cell_)), glm::ivec3(1));

    const uint32_t nc = cells();
    const float inv = 1.0f / cell_;
    const auto& pr = set.pos_rad();

    // The cell span of surfel i's bounding sphere, clamped to the grid.
    auto span = [&](uint32_t i, glm::ivec3& lo, glm::ivec3& hi) {
        const glm::vec3 c = glm::vec3(pr[i]);
        const float     s = pr[i].w;
        lo = glm::clamp(glm::ivec3(glm::floor((c - glm::vec3(s) - min_) * inv)),
                        glm::ivec3(0), res_ - 1);
        hi = glm::clamp(glm::ivec3(glm::floor((c + glm::vec3(s) - min_) * inv)),
                        glm::ivec3(0), res_ - 1);
    };
    auto index = [&](int x, int y, int z) {
        return uint32_t(x) + uint32_t(res_.x) * (uint32_t(y) + uint32_t(res_.y) * uint32_t(z));
    };

    // Counting sort. Pass 1 counts, pass 2 scatters into the prefix-summed
    // offsets. Both passes walk the same spans, so they cannot disagree.
    std::vector<uint32_t> count(nc + 1, 0);
    glm::ivec3 lo, hi;
    for (uint32_t i = 0; i < count_; ++i) {
        span(i, lo, hi);
        for (int z = lo.z; z <= hi.z; ++z)
            for (int y = lo.y; y <= hi.y; ++y)
                for (int x = lo.x; x <= hi.x; ++x) ++count[index(x, y, z)];
    }

    std::vector<glm::uvec2> sc(nc);
    uint32_t running = 0;
    occupied_ = 0;
    max_per_cell_ = 0;
    for (uint32_t c = 0; c < nc; ++c) {
        // The owner count rides in the high half, so a cell may hold at most
        // 65535 entries. This bake tops out at 9.
        assert(count[c] < 65536u);
        sc[c] = glm::uvec2(running, count[c]);
        if (count[c] != 0) ++occupied_;
        max_per_cell_ = std::max(max_per_cell_, count[c]);
        running += count[c];
    }
    entries_ = running;

    std::vector<uint32_t>  item(entries_);
    std::vector<glm::vec4> cpr(entries_);
    std::vector<uint32_t>  cursor(nc, 0);
    // The entry in the surfel's OWN cell carries kCellOwner. A range query wants
    // every entry -- that is what fat insertion is for -- but a volume query
    // (the NEE shadow march) walks a region and would otherwise meet the same
    // surfel once per cell it occupies, ~12 times in this bake, and pay a
    // projection for each. Marking one lets that query take one.
    //
    // Exactly one entry per surfel gets the bit: the centre cell is inside the
    // span by construction, since floor((c-s)/h) <= floor(c/h) <= floor((c+s)/h)
    // and both ends are clamped to the same grid.
    // Two sweeps, owners first, so a cell's owner entries are CONTIGUOUS at its
    // start and the count of them fits in the high half of sc.y. The volume
    // query then reads only those: it was walking every entry of every cell it
    // visited to test one bit, about 11500 four-byte reads per pixel to find 881
    // owners, and reads are what this pass is bound by.
    uint32_t owners = 0;
    for (int sweep = 0; sweep < 2; ++sweep) {
        for (uint32_t i = 0; i < count_; ++i) {
            span(i, lo, hi);
            const glm::ivec3 own = glm::clamp(
                glm::ivec3(glm::floor((glm::vec3(pr[i]) - min_) * inv)),
                glm::ivec3(0), res_ - 1);
            for (int z = lo.z; z <= hi.z; ++z)
                for (int y = lo.y; y <= hi.y; ++y)
                    for (int x = lo.x; x <= hi.x; ++x) {
                        const bool is_own = (x == own.x && y == own.y && z == own.z);
                        if (is_own != (sweep == 0)) continue;
                        const uint32_t c = index(x, y, z);
                        const uint32_t s = sc[c].x + cursor[c]++;
                        if (is_own) { ++owners; sc[c].y += 1u << 16; }
                        item[s] = i | (is_own ? 0x80000000u : 0u);
                        cpr[s]  = pr[i];
                    }
        }
    }
    // Cheap and load-bearing: a surfel with no owner entry is invisible to the
    // shadow march, and a surfel with two is counted twice.
    assert(owners == count_);

    // Coarse occupancy: one bit per 4x4x4 block.
    const int kMacro = 4;
    macro_res_ = (res_ + (kMacro - 1)) / kMacro;
    const uint32_t nmb = uint32_t(macro_res_.x) * uint32_t(macro_res_.y) *
                         uint32_t(macro_res_.z);
    std::vector<uint32_t> macro((nmb + 31u) / 32u, 0u);
    // ...and one bit per CELL inside each block, so a walk that has accepted a
    // block can iterate its occupied cells rather than testing all 64. The cone
    // march was running the full 4x4x4 for every block it accepted -- 16064
    // sphere-vs-cone tests per pixel to find 2683 cells worth reading (finding
    // 41). Bit c of block mb is cell (mb*4 + (c&3, (c>>2)&3, c>>4)).
    std::vector<glm::uvec2> macro_cell(nmb, glm::uvec2(0u));
    macro_occupied_ = 0;
    for (int z = 0; z < res_.z; ++z)
        for (int y = 0; y < res_.y; ++y)
            for (int x = 0; x < res_.x; ++x) {
                if (count[index(x, y, z)] == 0) continue;
                const uint32_t mb = uint32_t(x / kMacro) +
                                    uint32_t(macro_res_.x) *
                                    (uint32_t(y / kMacro) +
                                     uint32_t(macro_res_.y) * uint32_t(z / kMacro));
                if ((macro[mb >> 5] & (1u << (mb & 31u))) == 0u) ++macro_occupied_;
                macro[mb >> 5] |= 1u << (mb & 31u);
                const uint32_t c = uint32_t(x % kMacro) +
                                   4u * uint32_t(y % kMacro) +
                                   16u * uint32_t(z % kMacro);
                if (c < 32u) macro_cell[mb].x |= 1u << c;
                else         macro_cell[mb].y |= 1u << (c - 32u);
            }
    b_macro_.data(macro.data(), macro.size() * sizeof(uint32_t));
    b_macro_cell_.data(macro_cell.data(), macro_cell.size() * sizeof(glm::uvec2));

    b_sc_.data(sc.data(), sc.size() * sizeof(glm::uvec2));
    b_item_.data(item.data(), item.size() * sizeof(uint32_t));
    b_pr_.data(cpr.data(), cpr.size() * sizeof(glm::vec4));

    build_seconds_ = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0).count();

    gllib::logf(gllib::LogLevel::info,
        "grid: %dx%dx%d = %u cells (%u occupied, %.1f%%), cell %.4f (%.2f spacings), "
        "%u entries = %.2f per surfel, %.1f per occupied cell, max %u, %.1f KB, %.3f s, macro %dx%dx%d (%u occupied)",
        res_.x, res_.y, res_.z, nc, occupied_, 100.0 * double(occupied_) / double(nc),
        cell_, cell_ / set.spacing(), entries_, double(entries_) / double(count_),
        mean_per_occupied(), max_per_cell_, bytes() / 1024.0, build_seconds_,
        macro_res_.x, macro_res_.y, macro_res_.z, macro_occupied_);
}

void SurfelGrid::bind() const {
    b_sc_.bind_base(kBindCellSC);
    b_item_.bind_base(kBindCellItem);
    b_pr_.bind_base(kBindCellPR);
    b_macro_.bind_base(kBindMacro);
    b_macro_cell_.bind_base(kBindMacroCell);
}

std::size_t SurfelGrid::bytes() const {
    return std::size_t(cells()) * sizeof(glm::uvec2)
         + std::size_t(entries_) * (sizeof(uint32_t) + sizeof(glm::vec4));
}

} // namespace sgi
