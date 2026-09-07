#pragma once

// ---------------------------------------------------------------------------
// Spec section 9 step 0 -- the analytic gates.
//
// "Before anything else, run the single-surfel identity as a unit test... It
// catches every SH normalization convention error at once... If you are off by
// pi, 3.14x, or 9.87x, stop and fix it -- this error is CONSISTENT across both
// the near and far paths, so it will not show up as a discrepancy between them,
// only against an external reference."
//
// Driven by SGI_GATE, the way example 39 drives MOSAIC_SH_TEST -- no test
// framework, but they need a GL context, so main() runs them after init and
// exits before the frame loop.
// ---------------------------------------------------------------------------

#include "brute.hpp"

#include <string>

namespace sgi {

// `which` is a comma-separated list of gate names, or "all".
// Names: bake, p2p, disc, hemi, scale, noocc, occ, opaque, tonemap
// Returns true when every gate that ran passed.
bool run_gates(const std::string& which, SurfelSet& scene, Solver& solver,
               const SolveConfig& base, const std::vector<Tri>& tris);

} // namespace sgi
