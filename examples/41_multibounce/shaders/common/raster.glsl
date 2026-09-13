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

vec3 g_P, g_T, g_B, g_N;      // the camera, set once per workgroup

vec3 mbg_to_local(vec3 w) { return vec3(dot(w, g_T), dot(w, g_B), dot(w, g_N)); }
vec3 mbg_to_world(vec3 l) { return g_T * l.x + g_B * l.y + g_N * l.z; }

// Fill one straight-edged triangle of the projected polygon.
//
// Depth is EVALUATED, not interpolated: for the texel's direction w and the
// triangle's plane (pn, pc) in camera-local coordinates the hit parameter is
// exactly pc / dot(pn, w). That is the same cost as setting up an interpolant
// and it removes a whole class of precision question from a reference renderer.
void mbg_fill(vec2 A, vec2 B, vec2 C, vec3 pn, float pc, uint ti) {
    float area2 = (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);
    if (abs(area2) < 1e-14) return;
    float s = area2 < 0.0 ? -1.0 : 1.0;   // orientation, so one test works for both

    // THE FOLD TOLERANCE. A texel centre can land EXACTLY on an octahedral fold:
    // at an even target edge, texel (i,i) and texel (i, res-1-i) have t.x or t.y
    // identically zero. A triangle spanning that fold is split there into two
    // pieces whose shared edge passes exactly through the centre, so both edge
    // functions evaluate to zero plus float noise -- and if both pieces round the
    // wrong way, the texel is left empty and the camera sees a hole in a solid
    // wall.
    //
    // Measured before this tolerance existed: 3 empty texels in 65536 over 64
    // Cornell cameras, every one of them on a fold (the oracle gate prints them
    // with MBG_GATE_VERBOSE=1). Small, but it is a hole in a visibility buffer,
    // and the same geometry at a different orientation is free to make it a
    // bigger one.
    //
    // The fix is to let the two pieces OVERLAP by a hair instead of risking a
    // gap between them. atomicMin makes double coverage free -- both pieces
    // compute the same depth from the same plane -- so the asymmetry is entirely
    // in our favour. The tolerance scales with the piece's own area because the
    // edge functions are products of pixel coordinates and their noise floor
    // scales the same way.
    float eps = 1e-5 * abs(area2) + 1e-6;

    vec2 lo = min(A, min(B, C));
    vec2 hi = max(A, max(B, C));
    int res = int(u_res);
    int x0 = clamp(int(floor(lo.x)), 0, res - 1);
    int x1 = clamp(int(ceil (hi.x)), 0, res - 1);
    int y0 = clamp(int(floor(lo.y)), 0, res - 1);
    int y1 = clamp(int(ceil (hi.y)), 0, res - 1);

    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            vec2 p = vec2(float(x) + 0.5, float(y) + 0.5);
            // Inclusive on every edge, with the fold tolerance above. Shared
            // edges between fan triangles, and between the four quadrant pieces
            // of one triangle, are therefore covered twice -- which atomicMin
            // resolves to the same value. A top-left rule would avoid the double
            // write and open a one-texel crack instead, and a crack in a
            // visibility buffer is a light leak.
            float w0 = ((C.x - B.x) * (p.y - B.y) - (C.y - B.y) * (p.x - B.x)) * s;
            float w1 = ((A.x - C.x) * (p.y - C.y) - (A.y - C.y) * (p.x - C.x)) * s;
            float w2 = ((B.x - A.x) * (p.y - A.y) - (B.y - A.y) * (p.x - A.x)) * s;
            if (w0 < -eps || w1 < -eps || w2 < -eps) continue;

            vec3 w = mbg_px_to_dir(p, u_res);
            float denom = dot(pn, w);
            if (abs(denom) < 1e-20) continue;
            float t = pc / denom;
            if (t <= 0.0) continue;                    // behind the camera
            atomicMin(s_vis[uint(y) * u_res + uint(x)],
                      mbg_pack_key(t * length(w), u_inv_far, ti));
        }
    }
}

void mbg_raster_tri(uint ti) {
    MbgTri tr = tris[ti];

    vec3 v[MBG_CLIP_MAX];
    v[0] = mbg_to_local(tr.p0.xyz - g_P);
    v[1] = mbg_to_local(tr.p1.xyz - g_P);
    v[2] = mbg_to_local(tr.p2.xyz - g_P);

    vec3  pn = mbg_to_local(tr.n.xyz);   // rotation only: still unit, still the plane normal
    float pc = dot(pn, v[0]);

    // The horizon. The camera sits ON a surface, so its own triangle passes
    // through the origin; u_bias has already pushed the origin off that plane,
    // and this clip then removes it entirely along with everything below.
    vec3 hemi[MBG_CLIP_MAX];
    int nh = mbg_clip_plane(v, 3, vec3(0.0, 0.0, 1.0), hemi);
    if (nh < 3) return;

    for (int q = 0; q < 4; ++q) {
        float sx = (q & 1) == 0 ? 1.0 : -1.0;
        float sy = (q & 2) == 0 ? 1.0 : -1.0;

        vec3 cx[MBG_CLIP_MAX];
        int nx = mbg_clip_plane(hemi, nh, vec3(sx, 0.0, 0.0), cx);
        if (nx < 3) continue;
        vec3 cy[MBG_CLIP_MAX];
        int ny = mbg_clip_plane(cx, nx, vec3(0.0, sy, 0.0), cy);
        if (ny < 3) continue;

        vec2 px[MBG_CLIP_MAX];
        bool ok = true;
        for (int i = 0; i < ny; ++i) {
            float L;
            vec2 e = mbg_project(cy[i], sx, sy, L);
            // L <= 0 means the vertex is at or behind the centre of projection,
            // i.e. the geometry touches the camera origin. u_bias exists so this
            // does not happen; dropping the piece is the safe failure.
            if (!(L > 1e-12)) { ok = false; break; }
            px[i] = mbg_square_to_px(e, u_res);
        }
        if (!ok) continue;

        for (int k = 1; k + 1 < ny; ++k) mbg_fill(px[0], px[k], px[k + 1], pn, pc, ti);
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
// The rotation angle for a receiver at `pos`. `salt` separates independent
// sampling grids belonging to the same receiver -- the hemisphere and the sun's
// cone -- so that decorrelating one does not lock it to the other.
float mbg_jitter_angle(vec3 pos, uint salt) {
    uint seed = mbg_hash(salt ^ floatBitsToUint(pos.x) ^
                mbg_hash(floatBitsToUint(pos.y) ^
                mbg_hash(floatBitsToUint(pos.z))));
    return float(seed) * (1.0 / 4294967296.0) * 6.28318530717959;
}

void mbg_set_receiver(vec3 pos, vec3 nrm, float bias, bool jitter) {
    g_N = normalize(nrm);
    mbg_onb(g_N, g_T, g_B);
    g_P = pos + g_N * bias;

    if (jitter) {
        float a = mbg_jitter_angle(pos, 0u);
        float c = cos(a), sn = sin(a);
        vec3 t2 = g_T * c + g_B * sn;
        g_B = g_B * c - g_T * sn;
        g_T = t2;
    }
}

#endif
