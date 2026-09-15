#ifndef MBG_LIGHTVIEW_GLSL
#define MBG_LIGHTVIEW_GLSL

#include "counters.glsl"   // the probe below tallies into it

// ---------------------------------------------------------------------------
// The light view: a per-receiver perspective frustum fitted around one emitter,
// rasterized with the same atomicMin depth sort as everything else here.
//
// WHY THE HEMISPHERE CANNOT DO THIS JOB. Visibility off the hemisphere buffer is
// a ratio of texel masses, and its precision is however many texels the emitter
// covers. Cornell's panel subtends 0.36-2.8% of a receiver's hemisphere, which at
// a 32x32 target is between 4 and 29 texels -- so the visible fraction takes
// about five distinguishable values across a penumbra, and below one texel it
// collapses to a yes/no. That is not a soft shadow with a coarse ramp, it is a
// hard line with a couple of steps in it, and a Cornell box is made of the long
// soft penumbrae that shows up in.
//
// Raising the hemisphere's resolution is the wrong lever: 128x128 would spend
// 64 KB of shared memory to resolve a light occupying 1% of it. The right lever
// is to point the resolution AT the light. This frustum bounds the emitter's
// projection exactly, so a 16x16 target puts ~256 texels on the emitter instead
// of ~5, and the penumbra gets ~256 levels.
//
// It is still one mechanism: rasterize the scene from the shading point, let the
// depth sort resolve what is nearest, read the winner. The only thing that
// changes is which directions the texels are spent on. A triangle that does not
// project into the light's frustum is rejected by its bounding box and costs
// nothing, which is why this is CHEAPER than the hemisphere pass it replaces.
//
// Limitation: an emitter spanning more than about a hemisphere from the receiver
// has no bounded frustum. lv_setup returns false there and the caller treats the
// light as unoccluded -- right for a panel, wrong for a receiver inside a glowing
// box, which is what the hemisphere path is still there for.
// ---------------------------------------------------------------------------

#include "scene.glsl"
#include "hemi.glsl"

// NO SHARED VISIBILITY BUFFER. Inverting this rasterizer (finding 22) gave each
// texel to one thread, which keeps its winner in a register and needs no buffer
// at all -- but the two 1024-entry arrays the old atomicMin form used were left
// declared. 8 KB per workgroup, written by nothing and read by nothing, and on
// hardware shared memory is what caps blocks per SM. llvmpipe has no occupancy
// model, so it could not have shown this.

vec3 lv_P, lv_R, lv_U, lv_F;        // receiver origin and view frame
vec2 lv_lo, lv_span;                // the emitter's projected bounding box
float lv_near;
vec3 lv_N;                          // receiver normal, for culling
float lv_wmax;                      // the emitter's far extent along lv_F
// How close a triangle's PLANE has to pass to the receiver before "which side of
// it am I on" stops being a meaningful question. See lv_fill.
float lv_peps;
// Sub-texel offset of the sampling grid, in texels. Lets a caller break the
// alignment between a straight occluder edge and the grid's own rows.
vec2 lv_jit = vec2(0.0);

vec3 lv_project(vec3 d) { return vec3(dot(d, lv_R), dot(d, lv_U), dot(d, lv_F)); }

// Clip against w >= lv_near. That plane does not pass through the origin, so it
// cannot reuse mbg_clip_plane -- and it has to exist, because a wall running past
// the receiver straddles w = 0 and projects to infinity.
int lv_clip_near(vec3 src[MBG_CLIP_MAX], int n, out vec3 dst[MBG_CLIP_MAX]) {
    int m = 0;
    for (int i = 0; i < n; ++i) {
        vec3 a = src[i];
        vec3 b = src[i + 1 == n ? 0 : i + 1];
        float da = dot(a, lv_F) - lv_near;
        float db = dot(b, lv_F) - lv_near;
        if (da >= 0.0 && m < MBG_CLIP_MAX) dst[m++] = a;
        if ((da >= 0.0) != (db >= 0.0) && m < MBG_CLIP_MAX)
            dst[m++] = a + (b - a) * (da / (da - db));
    }
    return m;
}

// Aim the frustum at `tr` from `P` and fit it to the emitter. False when no
// bounded frustum exists.
bool lv_setup(vec3 P, vec3 N, MbgTri tr, float plane_eps) {
    lv_P = P;
    lv_N = N;
    lv_peps = plane_eps;
    vec3 c = (tr.p0.xyz + tr.p1.xyz + tr.p2.xyz) / 3.0 - P;
    float cl = length(c);
    if (cl < 1e-9) return false;
    lv_F = c / cl;
    lv_near = cl * 1e-3;

    vec3 up = abs(lv_F.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    lv_R = normalize(cross(up, lv_F));
    lv_U = cross(lv_F, lv_R);

    vec3 v[MBG_CLIP_MAX], cp[MBG_CLIP_MAX];
    v[0] = tr.p0.xyz - P;
    v[1] = tr.p1.xyz - P;
    v[2] = tr.p2.xyz - P;
    int n = lv_clip_near(v, 3, cp);
    if (n < 3) return false;

    vec2 lo = vec2(1e30), hi = vec2(-1e30);
    lv_wmax = 0.0;
    for (int i = 0; i < n; ++i) {
        vec3 q = lv_project(cp[i]);
        vec2 uv = q.xy / q.z;
        lo = min(lo, uv);
        hi = max(hi, uv);
        lv_wmax = max(lv_wmax, q.z);
    }
    // A hair of padding so the emitter's own silhouette sits inside the buffer
    // rather than exactly on its rim.
    vec2 pad = max((hi - lo) * 0.03, vec2(1e-6));
    lv_lo = lo - pad;
    lv_span = (hi + pad) - lv_lo;
    if (any(lessThanEqual(lv_span, vec2(0.0)))) return false;
    // tan(76 degrees) ~ 4: past that the light is most of the sky and a planar
    // projection of it stops being reasonable.
    return all(lessThan(abs(lv_lo), vec2(4.0))) && all(lessThan(lv_span, vec2(8.0)));
}

// The same frustum, aimed along a DIRECTION instead of fitted to a triangle:
// the sun, which is a disc at infinity and has no geometry to fit to.
//
// Two things fall away with the emitter. There is no bounding box to compute --
// the cone's half-extent is the sun's angular radius and nothing else -- and
// there is no far extent, because nothing in a scene is behind a light at
// infinity, so lv_wmax's cull is switched off rather than set to a distance.
//
// `near_d` is the near plane along the view direction, in world units. It exists
// for the same reason lv_setup's does: a wall running past the receiver straddles
// w = 0 and projects to infinity. The receiver's own triangle is removed earlier
// and more cheaply, by lv_raster_tri's receiver-plane cull.
void lv_setup_dir(vec3 P, vec3 N, vec3 dir, float tan_r, float near_d,
                  float plane_eps) {
    lv_P = P;
    lv_N = N;
    lv_peps = plane_eps;
    lv_F = normalize(dir);
    lv_near = near_d;

    vec3 up = abs(lv_F.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    lv_R = normalize(cross(up, lv_F));
    lv_U = cross(lv_F, lv_R);

    lv_lo = vec2(-tan_r);
    lv_span = vec2(2.0 * tan_r);
    lv_wmax = 3.4e38;            // no "beyond the light" to cull against
}

// The near plane every direction-aimed view uses, derived from the far distance
// the depth keys are already normalized by so that it needs no uniform of its
// own. Four orders of magnitude below the scene is far inside the camera bias.
float lv_near_for(float inv_far) { return 1e-4 / max(inv_far, 1e-30); }

// Spin the texel grid about its own axis.
//
// A DIRECTIONAL light needs this and an emitter does not, which is not obvious
// until it bites. lv_setup builds its frame from the direction to the emitter's
// centroid, so every receiver gets a different frame for free and the texel
// grid's quantization error is already decorrelated between neighbours. The sun
// is the same direction everywhere, so lv_setup_dir hands every receiver in the
// scene the IDENTICAL grid -- and an error that every receiver makes together is
// not noise, it is a pattern, exactly as in raster.glsl's mbg_set_receiver.
//
// Correlated error also does not average down through the recursion, which is
// why it stayed invisible for two bounces and then drew streaks across the
// ceiling at three (implementation.md, finding 17).
void lv_spin(float angle) {
    float c = cos(angle), s = sin(angle);
    vec3 r2 = lv_R * c + lv_U * s;
    lv_U = lv_U * c - lv_R * s;
    lv_R = r2;
}


// Texel centre -> unnormalized direction. |dir|^2 = 1 + u^2 + v^2.
vec3 lv_px_to_dir(vec2 px, uint res) {
    vec2 uv = lv_lo + px / float(res) * lv_span;
    return lv_F + uv.x * lv_R + uv.y * lv_U;
}

// --- The inverted path: one texel per thread, no shared buffer ---------------
//
// SAME DEPTH SORT, OPPOSITE LOOP ORDER. lv_raster_tri walks a triangle and
// atomicMins every texel it covers, so the parallelism is over TRIANGLES and the
// work inside each one is over texels, walked serially in a single lane. With 32
// triangles and 64 threads that means half the workgroup idles at the barrier
// while the critical path is whatever the largest triangle's bounding box costs
// one thread. Measured on an RTX 3060: the per-pixel direct pass scales almost
// linearly with texel count -- 806 ms at a 4x4 light view, 1967 at 8x8, 6495 at
// 16x16 -- which is that serial walk and nothing else.
//
// Inverting it gives each thread ONE TEXEL and asks, for that direction, which
// triangle is nearest. Perfectly balanced, and the winner lives in a register,
// so the shared visibility buffer, its atomics and the barriers around them all
// disappear.
//
// IT IS STILL RASTERIZATION, NOT A RAY CAST. lv_tri_hit below decides coverage
// with the same edge test a rasterizer uses, evaluated before the projection
// instead of after: the three planes through the receiver and the triangle's
// edges have normals cross(v_k, v_k+1), and a direction is inside the triangle's
// cone exactly when all three dot products share the sign of the triple product
// [v0,v1,v2]. After a projective map that expression IS the 2D edge function, up
// to a positive scale -- the same inclusive test, resolved by the same depth
// comparison. What changes is the order the loops are nested in.
//
// It also deletes the clipping. lv_raster_tri has to clip against the near plane
// and the four frustum sides before it can project, which is five
// Sutherland-Hodgman passes over dynamically indexed vec3[MBG_CLIP_MAX] arrays --
// arrays a GPU puts in local memory. Here there is nothing to clip: the only
// directions ever tested are texel centres, which are inside the frustum by
// construction, and the sign test handles the near plane on its own.
//
// The shape it leaves behind is the one culling wants. Every triangle is now an
// independent, side-effect-free test against a direction, so a future frustum or
// cluster cull just shortens the inner loop.

// Does triangle `ti` cover direction `d` from lv_P, and at what packed depth?
// MBG_EMPTY when it does not. `d` need not be normalized.
// --- Culling ----------------------------------------------------------------
//
// THE CULL HAS TO LIVE OUTSIDE THE TEXEL LOOP, which is the only subtle part.
// Inverting the rasterizer put texels on the outside and triangles on the
// inside, so a cull evaluated where the triangles are would be paid once per
// texel -- 4098 cluster tests times 64 texels, which is worse than not culling.
// The frustum is a property of the light view, not of the texel, so it is
// evaluated ONCE per view by the whole workgroup, and the survivors go in a
// shared list the texel loop then walks.
//
// This is why the inversion matters beyond its own speedup: the loop order it
// produced is the one culling attaches to.
#define MBG_MAX_VIS_CLUSTERS 256
shared uint s_cl[MBG_MAX_VIS_CLUSTERS];
shared uint s_cl_n;
shared uint s_cl_overflow;

// Is the box (relative to lv_P) anywhere inside the view's frustum?
//
// Six half-spaces: the four sides pass through the receiver so their offset is
// zero, plus near and far along lv_F. For a box, the extreme of dot(n, x) is
// dot(n, centre) +- dot(|n|, half-extent), which is why no corner is enumerated.
// The receiver's own plane is a seventh: anything entirely behind it cannot be
// between the receiver and anything else.
bool lv_aabb_visible(vec3 c, vec3 e) {
    vec2 hi2 = lv_lo + lv_span;
    vec3 n;
    n = lv_R - lv_lo.x * lv_F;   if (dot(n, c) + dot(abs(n), e) < 0.0) return false;
    n = hi2.x * lv_F - lv_R;     if (dot(n, c) + dot(abs(n), e) < 0.0) return false;
    n = lv_U - lv_lo.y * lv_F;   if (dot(n, c) + dot(abs(n), e) < 0.0) return false;
    n = hi2.y * lv_F - lv_U;     if (dot(n, c) + dot(abs(n), e) < 0.0) return false;
    float wc = dot(lv_F, c), we = dot(abs(lv_F), e);
    if (wc + we < lv_near)  return false;              // entirely behind the view
    if (wc - we > lv_wmax)  return false;              // entirely beyond the light
    if (dot(lv_N, c) + dot(abs(lv_N), e) < 0.0) return false;   // behind the receiver
    return true;
}

// Build the surviving-cluster list for the frustum lv_setup just established.
// Overflow is not an error: the list is abandoned and the texel loop walks
// everything, which is slower but exactly as correct.
void lv_cull(uint tid, uint stride, uint cluster_count) {
    // Below a handful of clusters the cull cannot pay for its own barriers, and
    // saying so costs one branch. Cornell is 32 triangles -- a SINGLE cluster --
    // and culling it made the sweep 1.8x slower before this early-out existed.
    if (u_cull == 0u || cluster_count <= 8u) {
        if (tid == 0u) { s_cl_n = 0u; s_cl_overflow = 1u; }
        barrier();
        return;
    }
    if (tid == 0u) { s_cl_n = 0u; s_cl_overflow = 0u; }
    barrier();
    for (uint c = tid; c < cluster_count; c += stride) {
        vec3 lo = clusters[c].lo.xyz - lv_P;
        vec3 hi = clusters[c].hi.xyz - lv_P;
        if (!lv_aabb_visible((lo + hi) * 0.5, (hi - lo) * 0.5)) continue;
        uint slot = atomicAdd(s_cl_n, 1u);
        if (slot < MBG_MAX_VIS_CLUSTERS) s_cl[slot] = c;
        else s_cl_overflow = 1u;
    }
    barrier();
}

uint lv_cl_count(uint cluster_count) {
    return s_cl_overflow != 0u ? cluster_count : min(s_cl_n, MBG_MAX_VIS_CLUSTERS);
}
uint lv_cl_at(uint k) { return s_cl_overflow != 0u ? k : s_cl[k]; }

// Does the ray from the light view's origin along `d` miss this cluster's box?
// Same slab test as the hemisphere's (finding 30), same NaN-falls-through-to-
// ACCEPT discipline: rejecting wrongly loses geometry and reads as a light leak.
bool lv_slab_miss(MbgCluster cl, vec3 d, vec3 inv) {
    // EXPANDED BY lv_peps, NOT BY AN EPSILON. lv_tri_hit is not purely a ray
    // test: finding 19's half-space rule reports a hit when the receiver lies
    // within lv_peps of a triangle's plane AND inside its bounds, whether or not
    // `d` geometrically passes through the triangle -- that rule is what stops
    // the sun leaking through concave creases. A box expanded only by a float
    // epsilon rejects those clusters and the leak comes back: measured at 92/255
    // on two pixels before this term was added. At lv_peps the rule can only
    // fire where the origin is already inside the expanded box, which the slab
    // test always accepts.
    vec3 e = (cl.hi.xyz - cl.lo.xyz) * 1e-5 + vec3(lv_peps + 1e-6);
    vec3 t1 = (cl.lo.xyz - e - lv_P) * inv;
    vec3 t2 = (cl.hi.xyz + e - lv_P) * inv;
    vec3 tlo = min(t1, t2), thi = max(t1, t2);
    float tn = max(max(tlo.x, tlo.y), max(tlo.z, 0.0));
    float tf = min(min(thi.x, thi.y), thi.z);
    return tn > tf;
}

bool lv_tri_hit(uint ti, vec3 d, out float dist) {
    // geom[], not mbg_tri(): this is the light view's inner loop and the shading
    // half of the record is never read here.
    MbgTriGeom tr = geom[ti];
    vec3 v0 = tr.p0.xyz - lv_P;
    vec3 v1 = tr.p1.xyz - lv_P;
    vec3 v2 = tr.p2.xyz - lv_P;

    // The same two culls lv_raster_tri opens with, for the same reasons: a
    // triangle wholly behind the receiver's own plane cannot be between it and
    // anything, and one wholly beyond the light cannot be in front of it.
    vec3 h = vec3(dot(lv_N, v0), dot(lv_N, v1), dot(lv_N, v2));
    if (all(lessThan(h, vec3(0.0)))) return false;
    vec3 w = vec3(dot(v0, lv_F), dot(v1, lv_F), dot(v2, lv_F));
    if (all(greaterThan(w, vec3(lv_wmax)))) return false;

    vec3  pn = tr.n.xyz;
    float pc = dot(pn, v0);

    // The receiver is ON this triangle's plane: everything on the far side of it
    // is inside the material. Finding 19 -- without this the sun leaks through
    // concave creases where the reconstructed position lands microns outside.
    if (abs(pc) <= lv_peps) {
        vec3 lo = min(v0, min(v1, v2)) - vec3(lv_peps);
        vec3 hi = max(v0, max(v1, v2)) + vec3(lv_peps);
        if (all(lessThanEqual(lo, vec3(0.0))) && all(greaterThanEqual(hi, vec3(0.0)))) {
            if (dot(pn, d) >= 0.0) return false;   // the open side: nothing there
            dist = 0.0;                            // nearest possible
            return true;
        }
    }

    if (!mbg_cone_contains(v0, v1, v2, d)) return false;

    float den = dot(pn, d);
    if (abs(den) < 1e-20) return false;
    float t = pc / den;
    if (t <= 0.0) return false;                  // behind the receiver
    // The ray parameter, not a distance: everything compared here shares this
    // same d, so the |d| scale is common and cancels.
    dist = t;
    return true;
}

// (visible, total) over this thread's texels, computing the visibility as it
// goes. Replaces clear + two rasterizations + lv_mass with one pass.
//
// The emitter is tested FIRST and the scene loop is skipped when it does not
// reach this texel, which the rasterized version could not do: it had to fill
// the whole scene into the buffer before it knew.
vec2 lv_mass_texel(uint res, vec3 N, uint emit_ti, uint cluster_count,
                   uint lo, uint stride) {
    vec2 m = vec2(0.0);
    uint n = res * res;
    for (uint i = lo; i < n; i += stride) {
        vec3 d = lv_px_to_dir(vec2(float(i % res), float(i / res)) + vec2(0.5) + lv_jit, res);
        float cosr = dot(N, d);
        if (cosr <= 0.0) continue;

        float de;
        if (!lv_tri_hit(emit_ti, d, de)) continue;

        // ANY BLOCKER, NOT THE NEAREST ONE. The question is whether this texel
        // sees the emitter, so the search can stop at the first triangle in
        // front of it rather than reducing over the whole scene. Identical
        // answer, and it turns the inner loop from a min into a search that
        // exits early exactly where the work is heaviest -- the shadowed texels.
        //
        // The threshold carries the relative tolerance that replaced the packed
        // key's one-quantum slack: an emitter coplanar with a surface (Cornell's
        // panel lies in the ceiling) must not be shadowed by the surface it
        // sits in.
        float thresh = de * (1.0 - 1e-4);
        bool blocked = false;
        float dd;
        uint ncl = lv_cl_count(cluster_count);
        // Reciprocal once per texel, not once per texel per cluster.
        vec3 inv = 1.0 / d;
        for (uint k = 0u; k < ncl && !blocked; ++k) {
            MbgCluster cl = clusters[lv_cl_at(k)];
            uint first = uint(cl.lo.w), last = first + uint(cl.hi.w);
            if (u_count != 0u) { g_tally[MBG_CT_LV_PAIR] += 1u;
                                 if (lv_slab_miss(cl, d, inv)) ++g_tally[MBG_CT_LV_ANG]; }
            if (u_lv_angular != 0u && lv_slab_miss(cl, d, inv)) continue;
            for (uint t = first; t < last; ++t)
                if (lv_tri_hit(t, d, dd) && dd < thresh) { blocked = true; break; }
        }

        float l2 = dot(d, d);
        float w = cosr / (l2 * l2);
        m.y += w;
        if (!blocked) m.x += w;
    }
    return m;
}

// The sun's disc, same inversion. It needs only "is anything in the way", not
// which thing, so the scene loop stops at the first blocker.
vec2 lv_mass_disc_texel(uint res, vec3 N, float cos_r, uint cluster_count,
                        uint lo, uint stride) {
    vec2 m = vec2(0.0);
    uint n = res * res;
    for (uint i = lo; i < n; i += stride) {
        vec3 d = lv_px_to_dir(vec2(float(i % res), float(i / res)) + vec2(0.5) + lv_jit, res);
        float l = length(d);
        if (dot(d, lv_F) < cos_r * l) continue;
        float cosr = dot(N, d);
        if (cosr <= 0.0) continue;

        bool blocked = false;
        float dd;
        uint ncl = lv_cl_count(cluster_count);
        vec3 inv = 1.0 / d;
        for (uint k = 0u; k < ncl && !blocked; ++k) {
            MbgCluster cl = clusters[lv_cl_at(k)];
            uint first = uint(cl.lo.w), last = first + uint(cl.hi.w);
            if (u_count != 0u) { g_tally[MBG_CT_LV_PAIR] += 1u;
                                 if (lv_slab_miss(cl, d, inv)) ++g_tally[MBG_CT_LV_ANG]; }
            if (u_lv_angular != 0u && lv_slab_miss(cl, d, inv)) continue;
            for (uint t = first; t < last; ++t)
                if (lv_tri_hit(t, d, dd)) { blocked = true; break; }
        }

        float w = cosr / (l * l * l * l);
        m.y += w;
        if (!blocked) m.x += w;
    }
    return m;
}

#endif
