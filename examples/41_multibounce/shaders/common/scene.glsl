#ifndef MBG_SCENE_GLSL
#define MBG_SCENE_GLSL

// ---------------------------------------------------------------------------
// The data every pass shares: the triangle soup, the secondary cameras, and the
// per-texel quadrature table.
//
// There is no acceleration structure and no LOD here, by design. Design doc
// section 5 is about not generating sub-pixel triangles; this example is the
// correctness reference that section 5's machinery gets asserted against, so it
// rasterizes the full-resolution mesh and loops every triangle for every camera.
// Cornell is 32 triangles, which is exactly why it is the scene this runs on.
// ---------------------------------------------------------------------------

// AoS, 96 bytes. Positions are world space, pre-transformed on the host.
//   n.w    != 0  double-sided (emits and reflects from both faces)
struct MbgTri {
    vec4 p0;
    vec4 p1;
    vec4 p2;
    vec4 n;
    vec4 albedo;
    vec4 emission;   // emitted RADIANCE, KHR_materials_emissive_strength folded in
};

// A secondary camera: a point on a surface plus the normal its hemisphere is
// built around.
//   pos.w != 0  active. Inactive slots exist so that camera i of a level always
//               maps to a fixed slot (background pixels, blocks that saw
//               nothing), which keeps every index in this example a pure
//               function of the schedule rather than of a compaction.
struct MbgCam {
    vec4 pos;
    vec4 nrm;
};

// Buffers are declared by each pass rather than here: the same binding is a
// readonly camera array in the raster and a writeonly one in the placement pass,
// and GLSL has no way to say that once. The BINDING NUMBERS are the contract and
// they are fixed across the example:
//
//   0  MbgTri   triangles           readonly everywhere
//   1  MbgCam   cameras, this level  written by place.comp / raster.comp's spawn
//   2  vec4     quadrature table for this level's resolution
//   3  vec4     irradiance, this level
//   4  MbgCam   cameras, next level (the spawn target)
//   5  vec4     next level's per-block albedo weights, xyz, w = active
//   6  vec4     irradiance, next level (read by the gather)
//
// quad[i] = vec4(unit direction in the TANGENT frame, cosine-weighted solid
// angle of texel i). See scene.cpp::build_quadrature -- the weights are
// integrated, not point-sampled, and normalized so that the sum over the target
// is exactly PI. Doc section 8.2 milestone 1 singles this out: "getting
// solid-angle weighting wrong here poisons everything downstream and is very
// hard to spot later".
// Orthonormal basis around n. Duff et al. 2017, branchless and stable at both
// poles. Any basis works -- irradiance is rotation invariant about n -- but it
// has to be the SAME basis in the raster and in the resolve, which is why both
// call this rather than deriving their own.
void mbg_onb(vec3 n, out vec3 t, out vec3 b) {
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float c = n.x * n.y * a;
    t = vec3(1.0 + s * n.x * n.x * a, s * c, -s * n.x);
    b = vec3(c, s + n.y * n.y * a, -n.y);
}

#endif
