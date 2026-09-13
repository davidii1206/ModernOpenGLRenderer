#pragma once

// ---------------------------------------------------------------------------
// Analytic gates. Doc section 8.2's milestone 1 ("validates the integration math
// before any performance work") and milestone 3 ("compare against the hardware
// path pixel for pixel"), run on the real kernels.
//
// They exist because the two errors this technique is most likely to have --
// a wrong solid-angle weight and a missing or doubled /PI -- are both INVISIBLE
// in an image. They scale the result by a constant, which reads as "the exposure
// is a bit off" and survives any amount of looking at Cornell boxes.
//
//   quad     the quadrature table sums to 2*PI and PI before normalization
//   closed   a camera sealed inside an emitter of radiance L reads exactly PI*L
//   rect     a camera under a rectangle matches the analytic form factor
//   occ      an opaque blocker takes it to zero
//   oracle   the compute rasterizer vs a CPU ray cast, texel for texel
//   series   N camera levels in a closed box of albedo p read PI*L*(1+p+...+p^N-1)
//
// `names` is "all" or a comma-separated subset. Returns false if any assertion
// failed, so SGI-style scripted runs can gate on the exit code.
// ---------------------------------------------------------------------------

#include "scene.hpp"
#include "solver.hpp"

#include <string>

namespace mbg {

bool run_gates(const std::string& names, Solver& solver, const SolveConfig& base_in,
               const Scene& scene, const std::vector<Tri>& tris);

} // namespace mbg
