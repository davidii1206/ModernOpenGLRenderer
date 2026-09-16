#pragma once

// ---------------------------------------------------------------------------
// Gate 11: does the deterministic direction assignment tile a parent's bins?
//
// world-space-radiance-cascades.md section 2.4 gives every c0 probe a fixed
// budget of D0 rays and assigns each one a sub-bin at cascade n from base-4
// digits of the probe's own cell offset along the two TANGENT axes of its normal
// class. The claim is that the children of a cascade-n probe then cover all of
// its D0 * 4^n bins with no randomness anywhere.
//
// That claim is the single point of failure for the whole architecture, because
// finding 49 removed its fallback: tracing cascades as separate shells costs
// more than the unbounded hemisphere, so there is no cheaper way to build the
// ladder if the assignment does not tile.
//
// It is also combinatorics, not rendering. A c0 probe sends the same digit tuple
// to every one of its D0 directions, so the covered fraction of a parent's bins
// is just how many DISTINCT tangent-plane digit tuples its descendants occupy --
// no tracing, no intervals, no merge, no storage. That is what this measures,
// from a real G-buffer, so the clutter and silhouettes are the real ones.
//
// Two ways it fails, and they are the interesting output:
//   - sparsity: fewer than 4^n descendants exist at all (silhouettes, occlusion)
//   - collision: the normal axis is ignored, so two probes at different depths
//     under one parent land on the same digit tuple. Exactly what a surface that
//     is not aligned with its class produces.
// ---------------------------------------------------------------------------

#include "screen.hpp"

#include <glm/glm.hpp>

namespace mbg {

// Reads the G-buffer back and reports the empty-bin fraction per cascade.
// Returns false if the readback failed or the frame held no geometry.
bool run_coverage(const GBuffer& gb, const glm::mat4& inv_view_proj,
                  const glm::vec3& cam_pos, float scene_diagonal);

} // namespace mbg
