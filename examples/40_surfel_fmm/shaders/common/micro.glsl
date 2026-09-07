#ifndef SGI_MICRO_GLSL
#define SGI_MICRO_GLSL

// ---------------------------------------------------------------------------
// The hemi-octahedral microbuffer.
//
// TEXELS ARE NOT EQUAL SOLID ANGLE. The square -> octahedron unfold is
// area-preserving, but octahedron -> sphere is a central projection whose
// Jacobian goes as 1/|p|^3, and |p| runs from 1/sqrt(3) at a face centre to 1 at
// a vertex. Measured over the actual mapping the per-texel solid angle varies by
// 3.2x at 8x8 and 4.2x at 16x16.
//
// Assuming a uniform 2*PI/MS^2 makes SUM(cos * dw) come out at 0.983*PI (8x8) or
// 0.969*PI (16x16) instead of PI -- a systematic 1.7-3.1% deficit that gets
// WORSE with more buckets. Example 39 makes this assumption and reports the
// resulting 1.6% as "16x16 quadrature error"; spec section 3.4 says to
// precompute bucketDir[] and bucketSolidAngle[], and it is right.
//
// So the host builds an exact table (build_bucket_table in brute.cpp):
//
//   bucket[2*b + 0] = vec4(dir.xyz in the TANGENT frame, dw)
//   bucket[2*b + 1] = vec4(wcos, 0, 0, 0)
//
// where dw is the texel's exact solid angle by Girard's theorem and wcos is
// INTEGRAL(cos(theta) dOmega) over the texel. Storing wcos rather than
// dw * dir.z matters: SUM(dw * dir.z) is off by 1.04% at 8x8, SUM(wcos) is PI to
// seven digits.
//
// The forward map direction -> texel is still exact and uniform, because the
// grid in the unit square is uniform whatever the solid angles are.
// ---------------------------------------------------------------------------

#include "oct.glsl"

// Tangent-frame direction -> bucket index. This is spec section 3.4's
// dirToBucket, with the fold in oct.glsl's hemi_oct_encode.
//
// Note the spec's warning about "an explicit 8-case neighbourBucket() against
// the folded coordinates" does NOT apply to this map. Setting d.z = 0 gives
// |d.x| + |d.y| = 1, which the 45-degree rotation sends to the boundary of the
// unit square: stepping off the edge is stepping BELOW THE HORIZON, not into a
// mirrored fold, and there is nothing to wrap to. That warning belongs to the
// sphere map in oct.glsl, which this is not. So a splat's footprint is simply
// clipped at the square's edge -- and its normalization sum must be taken over
// the UNCLIPPED box, so sub-horizon weight is dropped rather than piled onto
// the rim.
ivec2 micro_texel(vec3 local, uint ms) {
    vec2 e  = hemi_oct_encode(local);
    vec2 uv = (e * 0.5 + 0.5) * float(ms);
    return clamp(ivec2(uv), ivec2(0), ivec2(int(ms) - 1));
}

// Footprint radius in texels for a source of solid angle `omega` landing on a
// texel of solid angle `dw`.
//
// A disc of radius rt texels covers PI*rt^2 texels, hence PI*rt^2*dw steradians,
// so rt = sqrt(omega / (PI*dw)). Sizing it as (angular radius) / (texel angular
// radius) instead -- which is the intuitive move, and what example 39 does --
// double-counts a PI and comes out sqrt(PI) = 1.77x too wide. Renormalizing by
// the kernel sum keeps the ENERGY right either way, so the error shows up only
// as over-softened contact shadows, which is exactly the kind of thing that gets
// blamed on the algorithm.
float micro_footprint(float omega, float dw) {
    return sqrt(omega / (3.14159265358979 * max(dw, 1e-12)));
}

// --- the receiver's tangent frame -------------------------------------------
//
// Shared because two kernels must agree on it EXACTLY. bf_micro.comp builds a
// microbuffer in this frame; mb_reproject.comp turns stored bucket indices back
// into world directions and must rebuild the identical basis from the surfel
// index alone. A transposed or rotated basis still produces a plausible image,
// so a mismatch here is invisible by inspection -- gate `mbident` exists for it.

uint sgi_hash_u32(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Branchless orthonormal basis (Duff et al. 2017). Example 39's
// `abs(n.y) < 0.999` branch flips the frame discontinuously near the pole, which
// puts a seam across a curved surface; this does not.
void sgi_onb(vec3 n, out vec3 t, out vec3 b) {
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float c = n.x * n.y * a;
    t = vec3(1.0 + s * n.x * n.x * a, s * c, -s * n.x);
    b = vec3(c, s + n.y * n.y * a, -n.y);
}

// The frame surfel `i` built its microbuffer in. `rotate`: 0 none, 1 static per
// surfel, 2 per surfel per frame. Mode 1 is deterministic across frames and runs
// but decorrelated between neighbours, so the coherent octahedral banding on a
// flat wall becomes spatial noise instead of a pattern. Mode 2 adds the frame and
// only pays off with a temporal blend, which this example has none of.
//
// A STORED microbuffer can only be reprojected if its frame is reproducible, so
// mode 2 is incompatible with the reprojection path -- the reader would rebuild a
// different basis than the writer used. Asserted on the host.
void sgi_surfel_frame(uint i, vec3 n, uint rotate, uint frame, out vec3 T, out vec3 B) {
    sgi_onb(n, T, B);
    if (rotate != 0u) {
        uint seed = i * 2654435761u + (rotate == 2u ? frame : 0u);
        float a = float(sgi_hash_u32(seed)) * (1.0 / 4294967296.0) * 6.28318530717959;   // TWO_PI, literal so micro.glsl needs no include of surfel.glsl
        float c = cos(a), s = sin(a);
        vec3 T2 = T * c + B * s;
        B = B * c - T * s;
        T = T2;
    }
}

#endif
