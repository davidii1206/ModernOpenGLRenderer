#pragma once

// ---------------------------------------------------------------------------
// Per-surfel edge CUT PLANES, ported from example 38's `compute_surfel_cuts`.
//
// WHY THIS IS THE FIX FOR FINDING 26. A surface tiled with discs has two
// opposite errors and they cannot both be fixed by choosing a radius:
//
//   * in the INTERIOR the discs must overlap or the surface has holes and light
//     leaks through them (finding 25: 17% straight through a wall at the bake's
//     coverage of 1.0, which `nee_occ` fixes by inflating the visibility radius);
//   * at a BOUNDARY the discs must not overlap, because the union of discs whose
//     centres lie inside a surface extends up to r PAST its edge. Every
//     silhouette is dilated by one surfel radius, every shadow is slightly too
//     large, and a straight edge made of a row of circles comes out scalloped
//     with period equal to the surfel spacing. That is the row of teeth along
//     the tall box's contact line, and inflating the radius makes it worse.
//
// A cut plane resolves both at once: inflate freely, then clip the disc against
// the mesh feature edges it must not cross. Interior discs have no cuts and are
// unaffected; boundary discs end exactly on the edge.
//
// Cuts come from TOPOLOGY, not from neighbouring samples, so the planes land on
// triangle borders regardless of how the surface was sampled. Which side of an
// edge a disc keeps is decided from the incident faces' CENTROIDS, never from
// the surfel centre or the sign of an offset -- example 38 records that a
// barycentric lattice puts whole rows of samples exactly ON the border, where
// those quantities are zero and sign-based decisions flip for half of them.
// This example samples with an R2 sequence rather than a lattice, so exact zeros
// are rarer, but "rarer" is not a reason to use a test that has no defined
// answer at zero.
//
// DEVIATION FROM 38: the plane is stored packed, vec4(n, -dot(anchor, n)), one
// vec4 per cut, and the shader test is dot(p, n) + d > 0. Example 38 keeps the
// anchor form as two vec4s because its layout is "ready for per-frame skinned
// cuts" (skin n like a normal, the anchor like a point). This example's set is
// static -- there is no skinning path to keep a door open for -- so the packed
// form is exactly equivalent and halves the buffer.
// ---------------------------------------------------------------------------

#include "surfels.hpp"

#include <glm/glm.hpp>
#include <vector>

namespace sgi {

// Up to this many planes per surfel. A Cornell surfel needs at most 2 (it sits
// in a corner of a box face); 4 leaves room for a wedge and the bake reports how
// many surfels saturate, which is the signal that it is too small.
constexpr int kSurfelCuts = 4;

// A sharp or boundary mesh edge in world space, with everything the cut
// derivation needs about its incident faces.
struct FeatEdge {
    glm::vec3 mid{0.0f};
    glm::vec3 dir{0.0f};          // unit
    float     half_len = 0.0f;
    glm::vec3 na{0.0f}, nb{0.0f}; // incident face normals (nb unused if boundary)
    glm::vec3 ctr_a{0.0f}, ctr_b{0.0f};
    glm::vec3 tri_a[3]{};         // face a's triangle, for the boundary ownership test
    int       obj_a = 0, obj_b = 0;
    bool      boundary = false;
};

struct FeatureEdges {
    std::vector<FeatEdge> edges;
    std::vector<int>      tri_object;      // per triangle, connected-component id
    std::vector<float>    tri_curvature;   // per triangle, from smooth dihedrals
};

// Sharp creases and open boundaries, with per-triangle object ids (union-find
// over shared edges) and curvature. Both gate the cuts: an edge only ever cuts
// surfels of its OWN object, so a box standing on the floor is not clipped by
// the floor; and cuts are only taken between FLAT faces, because a plane through
// a crease on a curved surface slices unrelated geometry.
FeatureEdges extract_feature_edges(const std::vector<Tri>& tris);

// One vec4 per cut slot, kSurfelCuts per surfel, packed as (n, -dot(anchor, n)).
// Empty slots are zero, which never rejects. `tri_of` is the home triangle of
// each surfel, as recorded by the bake.
std::vector<glm::vec4> compute_surfel_cuts(const SurfelSet& set,
                                           const std::vector<uint32_t>& tri_of,
                                           const std::vector<Tri>& tris,
                                           const FeatureEdges& fe);

enum CutBinding : uint32_t {
    kBindCuts = 22,   // vec4[N * kSurfelCuts], packed planes (n, -dot(anchor, n))
};

// The GPU-side cut buffer. Empty is a valid state and costs nothing: the shader
// reads slot 0, finds n == 0, and takes the uncut path.
class CutSet {
public:
    void build(const SurfelSet& set, const std::vector<Tri>& tris);
    void bind() const;

    bool     valid() const { return count_ != 0; }
    uint32_t count() const { return count_; }        // surfels covered
    std::size_t bytes() const;

private:
    gl::Buffer b_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
    uint32_t   count_ = 0;
};

} // namespace sgi
