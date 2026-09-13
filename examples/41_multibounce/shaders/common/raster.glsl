#ifndef MBG_RASTER_GLSL
#define MBG_RASTER_GLSL

// ---------------------------------------------------------------------------
// The hemisphere rasterizer: one workgroup, one receiver, one LDS visibility
// buffer resolved by atomicMin.
//
// Shared by every pass that needs to know what a point can see -- the secondary
// cameras of the recursion (raster.comp) and the per-pixel direct term
// (direct_pixel.comp). There is exactly one visibility mechanism in this
// renderer and this is it: rasterize, let the depth sort decide, read the
// winner. No rays, no shadow maps, no second structure.
//
// The includer declares `tris[]` and `quad[]`; the two uniforms the rasterizer
// itself needs are declared here, because a GLSL header cannot use a uniform the
// includer declares after it. Set the receiver frame with mbg_set_receiver()
// before rasterizing.
// ---------------------------------------------------------------------------

#include "scene.glsl"
#include "hemi.glsl"

uniform uint  u_res;          // hemisphere target edge, texels

// The largest target any level may use. The default schedule needs 256, but the
// buffer is sized for the cap so MBG_RES can be raised without a recompile --
// and a level-3 camera with an 8x8 target reserves the same 4 KB either way,
// because the array is sized once for every level. A production version would
// compile a variant per tier; this one keeps a single kernel.
#define MBG_MAX_TEXELS 1024
shared uint s_vis[MBG_MAX_TEXELS];
shared vec4 s_red[64];
shared vec4 s_red2[64];

// The continuation distribution, for the single-sample path estimator: one
// weight per texel, turned into an inclusive CDF in place. Only raster.comp's
// spawn uses it, and only after the light views are finished, so it could alias
// s_lv -- but 4 KB of shared memory is cheaper than the class of bug that comes
// from two buffers sharing storage across a barrier, and the budget here is
// about 18 KB against a 32 KB floor.
shared float s_cdf[MBG_MAX_TEXELS];
shared float s_blk[64];

vec3 g_P, g_T, g_B, g_N;      // the camera, set once per workgroup

vec3 mbg_to_local(vec3 w) { return vec3(dot(w, g_T), dot(w, g_B), dot(w, g_N)); }
vec3 mbg_to_world(vec3 l) { return g_T * l.x + g_B * l.y + g_N * l.z; }

// --- The hemisphere, inverted ------------------------------------------------
//
// One texel per thread, each asking which triangle is nearest along its own
// direction, instead of one triangle per thread atomicMin-ing every texel it
// covers. Same depth sort, opposite loop order -- implementation.md finding 22,
// and lv_mass_texel in lightview.glsl for the version that came first.
//
// AND IT DELETES THE OCTAHEDRAL CLIPPING ENTIRELY. Everything this file argues
// about the folds -- that the map is only piecewise projective, that a triangle
// must be cut at x=0, y=0 and the horizon before its edges are straight, that
// the two pieces need an overlap tolerance or a fold texel falls through the
// crack -- is a consequence of PROJECTING. None of it is a property of the
// visibility question. A direction either lies in a triangle's cone or it does
// not, and mbg_cone_contains answers that without a projection, so there is
// nothing to clip and no fold to fall into. mbg_raster_tri and mbg_fill stay for
// the gates to compare against, but nothing in a frame calls them.
//
// s_vis still lives in shared memory, unlike the light view's buffer: the
// quadrature reads only its own texels, but the tent-weighted spawn reads a
// neighbourhood, so the winners have to be visible across the workgroup. What
// goes away is the atomics -- each texel is written once, by the thread that
// owns it.
bool mbg_tri_hit(uint ti, vec3 d, out float dist) {
    MbgTri tr = tris[ti];
    vec3 v0 = tr.p0.xyz - g_P;
    vec3 v1 = tr.p1.xyz - g_P;
    vec3 v2 = tr.p2.xyz - g_P;

    vec3  pn = tr.n.xyz;
    float pc = dot(pn, v0);

    // NO HALF-SPACE RULE HERE, unlike the light view's lv_texel_key.
    //
    // That rule (finding 19) is a deliberate departure from what a ray cast
    // reports: it declares the far side of a plane the receiver sits on to be
    // solid, because a receiver reconstructed from a depth buffer can land
    // microns outside a wall and a ray would then correctly find nothing in the
    // way. The light view needs it, because the sun's shadow is one direction
    // wide and a single leaked pixel is visible.
    //
    // The hemisphere must not have it, because the oracle gate holds this
    // buffer against a CPU ray cast texel for texel, and a rule that ray casting
    // does not share makes them disagree by construction -- 480 texels and half
    // the energy in dispute when it was in here. Nor does it need it: the
    // camera's own triangle is excluded already, because its plane sits `bias`
    // below the origin, so every upper-hemisphere direction gives t < 0.

    if (!mbg_cone_contains(v0, v1, v2, d)) return false;

    float den = dot(pn, d);
    if (abs(den) < 1e-20) return false;
    float t = pc / den;
    if (t <= 0.0) return false;                  // behind the camera
    // The ray parameter, not a distance: every triangle here is compared along
    // the SAME d, so the scale factor |d| is common and cancels.
    dist = t;
    return true;
}

// Fill s_vis for this thread's texels. Replaces clear + rasterize + barrier.
void mbg_resolve_vis(uint tid, uint stride, uint tri_count) {
    uint n = u_res * u_res;
    for (uint i = tid; i < n; i += stride) {
        vec3 d = mbg_to_world(mbg_px_to_dir(vec2(float(i % u_res), float(i / u_res))
                                            + vec2(0.5), u_res));
        uint  best = MBG_EMPTY;
        float bestd = 1e30;
        for (uint t = 0u; t < tri_count; ++t) {
            float dist;
            if (mbg_tri_hit(t, d, dist) && dist < bestd) { bestd = dist; best = t; }
        }
        s_vis[i] = best;
    }
}

// --- The direct term, analytically ------------------------------------------
//
// Doc section 3, non-goals: "No correct area-light soft shadows from the GI
// pass. Direct lighting stays a separate conventional pass." This is that pass,
// kept inside the same kernel because the visibility it needs is already
// sitting in shared memory.
//
// WHY IT IS NOT OPTIONAL HERE. The emitter is the brightest thing in the scene
// by two orders of magnitude and subtends a few percent of the hemisphere, so a
// texel either sees all of it or none of it. At 32x32 that is a few percent of
// error per camera, uncorrelated between neighbours, and it reads as the
// mottling that dominated the first version of this example. At the 8x8 targets
// the deeper levels use, the panel can miss every texel centre of a camera that
// plainly sees it, and the indirect term goes blotchy.
//
// So the emitter's contribution is split in two:
//
//   magnitude   the EXACT projected solid angle of the emitter polygon, clipped
//               to the hemisphere -- Lambert's formula, no quadrature at all
//   visibility  a contribution-weighted fraction, sampled against the depth
//               already in s_vis
//
// The magnitude carries all the energy and is now exact. The visibility is the
// only sampled quantity left, it is bounded in [0,1], and it is 1 or 0 over most
// of the image with a smooth ramp across penumbrae -- which is exactly the error
// budget an area light wants. The hemisphere resolve then skips L_e on these
// triangles (emission.w), so nothing is counted twice.

// Clear, then rasterize a triangle range. `stride` is the workgroup size.
void mbg_clear_vis(uint tid, uint stride) {
    uint texels = u_res * u_res;
    for (uint i = tid; i < texels; i += stride) s_vis[i] = MBG_EMPTY;
}

// 32-bit integer hash (Chris Wellons' triple32-style mix). Only used to pick a
// rotation, so the bar is "decorrelated between neighbours", not randomness.
uint mbg_hash(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// THE TANGENT FRAME IS ROTATED PER RECEIVER, AND THAT IS NOT A DETAIL.
//
// Every receiver sampling the hemisphere with the SAME texel grid means every
// receiver makes the same quadrature error, and an error that all of them make
// together is not noise -- it is a pattern. A scene feature crosses a texel
// boundary at the same place for a whole row of receivers and their sums step
// together, which is why the indirect term came out banded rather than grainy
// (implementation.md, finding 14).
//
// Rotating each receiver's frame by a hash of its own position decorrelates
// them: the same error becomes spatial noise, which anything that averages
// neighbours removes. Example 40 reaches the same conclusion about its own
// microbuffer in one line of common/micro.glsl -- "the coherent octahedral
// banding on a flat wall becomes spatial noise instead of a pattern" -- and is
// the reason it gets clean indirect out of 8x8 buckets.
//
// The hash is over the receiver's POSITION rather than its index, so the
// rotation does not change when the chunk scheduler assigns a camera to a
// different slot; a sweep is reproducible bit for bit either way.
// A uniform in [0,1) for a receiver at `pos`, on an independent stream.
//
// EVERY random number in this renderer comes from here, and every one of them is
// a hash of a POSITION rather than of a camera index or a frame counter. That is
// what keeps a sweep reproducible bit for bit: the chunk scheduler is free to
// assign a camera to a different slot, and a receiver at the same point still
// draws the same numbers. `stream` separates the decisions made at one point --
// the hemisphere's rotation, the sun cone's, which direction a path continues
// through, whether it survives roulette -- so that decorrelating one does not
// lock it to another.
float mbg_rand(vec3 pos, uint stream) {
    uint seed = mbg_hash(stream ^ floatBitsToUint(pos.x) ^
                mbg_hash(floatBitsToUint(pos.y) ^
                mbg_hash(floatBitsToUint(pos.z))));
    return float(seed) * (1.0 / 4294967296.0);
}

float mbg_jitter_angle(vec3 pos, uint salt) {
    return mbg_rand(pos, salt) * 6.28318530717959;
}

// --- The continuation distribution ------------------------------------------
//
// s_cdf holds a non-negative weight per texel on entry and its INCLUSIVE prefix
// sum on exit; the return value is the total, identical in every invocation.
//
// Blocked scan rather than a serial one on thread 0: 64 contiguous runs scanned
// in parallel, a Hillis-Steele scan of the 64 run totals, then one add per
// entry. At 256 texels that is 4 serial adds plus 6 barriers instead of 256
// dependent adds with 63 lanes idle.
float mbg_scan_cdf(uint tid, uint n) {
    uint per = (n + 63u) / 64u;
    uint lo = min(tid * per, n);
    uint hi = min(lo + per, n);

    float run = 0.0;
    for (uint i = lo; i < hi; ++i) { run += s_cdf[i]; s_cdf[i] = run; }
    s_blk[tid] = run;
    barrier();

    for (uint d = 1u; d < 64u; d <<= 1u) {
        float v = tid >= d ? s_blk[tid - d] : 0.0;
        barrier();
        s_blk[tid] += v;
        barrier();
    }
    // s_blk[tid] is the inclusive scan of the run totals, so subtracting this
    // run's own total leaves the offset the run starts at.
    float off = s_blk[tid] - run;
    for (uint i = lo; i < hi; ++i) s_cdf[i] += off;
    float total = s_blk[63];
    barrier();
    return total;
}

// The first index whose inclusive CDF reaches `u`. No barriers: the search is
// per invocation and reads only.
uint mbg_sample_cdf(float u, uint n) {
    uint lo = 0u, hi = n - 1u;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1u;
        if (s_cdf[mid] < u) lo = mid + 1u; else hi = mid;
    }
    return lo;
}

void mbg_set_receiver(vec3 pos, vec3 nrm, float bias, bool jitter) {
    g_N = normalize(nrm);
    mbg_onb(g_N, g_T, g_B);
    g_P = pos + g_N * bias;

    if (jitter) {
        float a = mbg_jitter_angle(pos, 0x27D4EB2Du);
        float c = cos(a), sn = sin(a);
        vec3 t2 = g_T * c + g_B * sn;
        g_B = g_B * c - g_T * sn;
        g_T = t2;
    }
}

#endif
