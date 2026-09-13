#ifndef MBG_LIGHTVIEW_GLSL
#define MBG_LIGHTVIEW_GLSL

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

#define MBG_LV_MAX 1024
shared uint s_lv[MBG_LV_MAX];       // everything: what is actually nearest
shared uint s_lv_e[MBG_LV_MAX];     // the emitter alone: where it would reach

vec3 lv_P, lv_R, lv_U, lv_F;        // receiver origin and view frame
vec2 lv_lo, lv_span;                // the emitter's projected bounding box
float lv_near;
vec3 lv_N;                          // receiver normal, for culling
float lv_wmax;                      // the emitter's far extent along lv_F
// How close a triangle's PLANE has to pass to the receiver before "which side of
// it am I on" stops being a meaningful question. See lv_fill.
float lv_peps;

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

vec2 lv_to_px(vec2 uv, uint res) { return (uv - lv_lo) / lv_span * float(res); }

// Texel centre -> unnormalized direction. |dir|^2 = 1 + u^2 + v^2.
vec3 lv_px_to_dir(vec2 px, uint res) {
    vec2 uv = lv_lo + px / float(res) * lv_span;
    return lv_F + uv.x * lv_R + uv.y * lv_U;
}

// Clip against the frustum's four SIDE planes, which -- unlike its near plane --
// all pass through the receiver, so they are great circles in direction space and
// mbg_clip_plane takes them unchanged. Normals point inward: u >= lv_lo.x is
// dot(d, lv_R) - lv_lo.x * dot(d, lv_F) >= 0, and so on round the rectangle. The
// scale of each normal is irrelevant; only the sign and the ratio are used.
//
// WHY THIS IS NOT AN OPTIMIZATION. A triangle outside the frustum still projects,
// and its pixel coordinates come out as large as the ratio between its angular
// extent and the frustum's. For an emitter that ratio is single digits. For the
// SUN -- a cone about a degree across -- it is several thousand, so a Cornell
// wall lands at pixel coordinates of order 1e4 on an 8x8 buffer. Two things then
// break at once: float32 edge functions built from products of those coordinates
// carry absolute noise comparable to lv_fill's area-relative tolerance, and the
// tolerance itself, being a fraction of the triangle's own area, becomes a
// fraction of a texel of dilation. Both make an occluder a little bigger than it
// is, which on a shadow whose entire penumbra is `res` texels wide is the
// difference between a soft edge and a smeared one. Clipping first bounds every
// coordinate by construction, and it is faster as well, since the bounding box
// that follows no longer has to be clamped down from thousands of texels.
int lv_clip_frustum(vec3 src[MBG_CLIP_MAX], int n, out vec3 dst[MBG_CLIP_MAX]) {
    vec2 hi = lv_lo + lv_span;
    vec3 a[MBG_CLIP_MAX], b[MBG_CLIP_MAX];
    int m = mbg_clip_plane(src, n, lv_R - lv_lo.x * lv_F, a);
    if (m < 3) return 0;
    int k = mbg_clip_plane(a, m, hi.x * lv_F - lv_R, b);
    if (k < 3) return 0;
    m = mbg_clip_plane(b, k, lv_U - lv_lo.y * lv_F, a);
    if (m < 3) return 0;
    return mbg_clip_plane(a, m, hi.y * lv_F - lv_U, dst);
}

void lv_fill(vec2 A, vec2 B, vec2 C, vec3 pn, float pc, uint ti, uint res,
             float inv_far, bool emit_pass) {
    float area2 = (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);
    if (abs(area2) < 1e-14) return;
    float s = area2 < 0.0 ? -1.0 : 1.0;
    float eps = 1e-5 * abs(area2) + 1e-6;

    vec2 lo = min(A, min(B, C));
    vec2 hi = max(A, max(B, C));
    int r = int(res);
    int x0 = clamp(int(floor(lo.x)), 0, r - 1);
    int x1 = clamp(int(ceil (hi.x)), 0, r - 1);
    int y0 = clamp(int(floor(lo.y)), 0, r - 1);
    int y1 = clamp(int(ceil (hi.y)), 0, r - 1);

    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            vec2 p = vec2(float(x) + 0.5, float(y) + 0.5);
            float w0 = ((C.x - B.x) * (p.y - B.y) - (C.y - B.y) * (p.x - B.x)) * s;
            float w1 = ((A.x - C.x) * (p.y - C.y) - (A.y - C.y) * (p.x - C.x)) * s;
            float w2 = ((B.x - A.x) * (p.y - A.y) - (B.y - A.y) * (p.x - A.x)) * s;
            if (w0 < -eps || w1 < -eps || w2 < -eps) continue;

            // Depth evaluated, not interpolated: the plane intersection along
            // this texel's own ray, exactly as the hemisphere rasterizer does it.
            vec3 d = lv_px_to_dir(p, res);
            float den = dot(pn, d);
            if (abs(den) < 1e-20) continue;
            float t = pc / den;
            if (t <= 0.0) {
                // A LIGHT LEAK AT EVERY CONCAVE CREASE.
                //
                // t <= 0 means the plane is behind the receiver, so the texel is
                // dropped. But a receiver in a corner sits ON the adjacent wall's
                // plane: pc is the distance to it, which is zero in exact
                // arithmetic and a few times 1e-6 of float noise in practice, so
                // its SIGN is decided by rounding. Half the time the wall the
                // receiver is touching is declared to be behind it and stops
                // occluding -- and then the sun, whose whole hemisphere on that
                // side is wall, comes through at full strength in a single pixel.
                // That is the white speckle that used to run down the red wall's
                // back corner.
                //
                // Coverage is the geometric truth here and the sign is not, so a
                // plane passing within the receiver's own placement tolerance is
                // treated as what it is: an occluder at zero distance, nearest
                // possible. Anything genuinely behind still goes.
                if (abs(pc) > lv_peps) continue;
                t = 0.0;
            }
            uint key = mbg_pack_key(t * length(d), inv_far, ti);
            uint idx = uint(y) * res + uint(x);
            if (emit_pass) atomicMin(s_lv_e[idx], key);
            else           atomicMin(s_lv[idx], key);
        }
    }
}

void lv_raster_tri(uint ti, uint res, float inv_far, bool emit_pass) {
    MbgTri tr = tris[ti];
    vec3 v[MBG_CLIP_MAX], cp[MBG_CLIP_MAX];
    v[0] = tr.p0.xyz - lv_P;
    v[1] = tr.p1.xyz - lv_P;
    v[2] = tr.p2.xyz - lv_P;

    // Two dot products a vertex, and they remove most of the scene. A triangle
    // entirely behind the receiver's own plane cannot be between it and
    // anything; a triangle entirely beyond the emitter cannot be in front of it.
    // Worth doing because this runs per pixel per emitter over every triangle in
    // the scene -- it is the loop that makes the pass expensive.
    vec3 h = vec3(dot(lv_N, v[0]), dot(lv_N, v[1]), dot(lv_N, v[2]));
    if (all(lessThan(h, vec3(0.0)))) return;
    vec3 w = vec3(dot(v[0], lv_F), dot(v[1], lv_F), dot(v[2], lv_F));
    if (all(greaterThan(w, vec3(lv_wmax)))) return;
    if (all(lessThan(w, vec3(lv_near)))) return;

    vec3 pn = tr.n.xyz;
    float pc = dot(pn, v[0]);

    // THE RECEIVER IS ON THIS TRIANGLE'S PLANE, AND PROJECTION CANNOT SETTLE IT.
    //
    // A shading point in a concave corner lies on the adjacent wall as well as on
    // its own, and the position it lies on that wall is decided by a depth
    // reconstruction: the pixel's depth, unprojected, plus a bias along the
    // pixel's own normal, which for a perpendicular wall moves it not at all. At
    // the one-pixel-wide column where the G-buffer flips from one wall to the
    // other, that reconstruction can land a few microns OUTSIDE the room -- and
    // from outside, the wall is behind the receiver, its projection misses the
    // frustum entirely, and the sun shines straight through. Three pixels down
    // the red wall's back corner, at full sun, in an image where everything
    // around them is in shadow.
    //
    // No amount of care in the rasterizer fixes that, because given the position
    // it was handed the rasterizer is right. What is wrong is the position, and
    // the robust statement is about the PLANE rather than about the projection:
    // a receiver within its own placement tolerance of a triangle's plane is on
    // that surface, and everything on the far side of it is inside the material.
    // So that half-space is marked occluded directly, at distance zero.
    //
    // The bounding-box test is what keeps this from being a licence to block:
    // "the plane passes near the receiver" is satisfied by any distant triangle
    // whose plane happens to sweep past, and only a triangle whose own extent
    // also contains the receiver is actually the surface it is standing on. The
    // receiver's own triangle never reaches here -- the plane cull above removes
    // it -- and if it did, the half-space it would mark is the one below the
    // horizon, which every consumer already discards.
    //
    // Only on the scene pass: the emitter pass rasterizes one triangle to find
    // where the light reaches, and a coplanar emitter contributes nothing at all
    // (common/emitter.glsl rejects it before any of this runs).
    if (!emit_pass && abs(pc) <= lv_peps) {
        vec3 lo = min(v[0], min(v[1], v[2])) - vec3(lv_peps);
        vec3 hi = max(v[0], max(v[1], v[2])) + vec3(lv_peps);
        if (all(lessThanEqual(lo, vec3(0.0))) && all(greaterThanEqual(hi, vec3(0.0)))) {
            uint key = mbg_pack_key(0.0, inv_far, ti);
            for (uint i = 0u; i < res * res; ++i) {
                vec3 d = lv_px_to_dir(vec2(float(i % res), float(i / res)) + vec2(0.5), res);
                if (dot(pn, d) < 0.0) atomicMin(s_lv[i], key);
            }
            return;
        }
    }

    int n = lv_clip_near(v, 3, cp);
    if (n < 3) return;
    vec3 cf[MBG_CLIP_MAX];
    n = lv_clip_frustum(cp, n, cf);
    if (n < 3) return;

    vec2 px[MBG_CLIP_MAX];
    for (int i = 0; i < n; ++i) {
        vec3 q = lv_project(cf[i]);
        px[i] = lv_to_px(q.xy / q.z, res);
    }
    for (int k = 1; k + 1 < n; ++k)
        lv_fill(px[0], px[k], px[k + 1], pn, pc, ti, res, inv_far, emit_pass);
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
// IT IS STILL RASTERIZATION, NOT A RAY CAST. lv_texel_key below decides coverage
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
uint lv_texel_key(uint ti, vec3 d, float inv_far) {
    MbgTri tr = tris[ti];
    vec3 v0 = tr.p0.xyz - lv_P;
    vec3 v1 = tr.p1.xyz - lv_P;
    vec3 v2 = tr.p2.xyz - lv_P;

    // The same two culls lv_raster_tri opens with, for the same reasons: a
    // triangle wholly behind the receiver's own plane cannot be between it and
    // anything, and one wholly beyond the light cannot be in front of it.
    vec3 h = vec3(dot(lv_N, v0), dot(lv_N, v1), dot(lv_N, v2));
    if (all(lessThan(h, vec3(0.0)))) return MBG_EMPTY;
    vec3 w = vec3(dot(v0, lv_F), dot(v1, lv_F), dot(v2, lv_F));
    if (all(greaterThan(w, vec3(lv_wmax)))) return MBG_EMPTY;

    vec3  pn = tr.n.xyz;
    float pc = dot(pn, v0);

    // The receiver is ON this triangle's plane: everything on the far side of it
    // is inside the material. Finding 19 -- without this the sun leaks through
    // concave creases where the reconstructed position lands microns outside.
    if (abs(pc) <= lv_peps) {
        vec3 lo = min(v0, min(v1, v2)) - vec3(lv_peps);
        vec3 hi = max(v0, max(v1, v2)) + vec3(lv_peps);
        if (all(lessThanEqual(lo, vec3(0.0))) && all(greaterThanEqual(hi, vec3(0.0))))
            return dot(pn, d) < 0.0 ? mbg_pack_key(0.0, inv_far, ti) : MBG_EMPTY;
    }

    // The three edge planes through the receiver. `o` is the triple product,
    // which is the solid angle's orientation; a triangle seen edge on has o ~ 0
    // and subtends nothing.
    vec3 e0 = cross(v0, v1);
    vec3 e1 = cross(v1, v2);
    vec3 e2 = cross(v2, v0);
    float o = dot(e0, v2);
    if (abs(o) < 1e-20) return MBG_EMPTY;
    float sgn = o < 0.0 ? -1.0 : 1.0;
    float b0 = dot(d, e0) * sgn;
    float b1 = dot(d, e1) * sgn;
    float b2 = dot(d, e2) * sgn;
    // INCLUSIVE, WITH A RELATIVE TOLERANCE, and it is not optional -- this was
    // written without one first and the `occ` gate caught it.
    //
    // A texel direction can land EXACTLY on the edge two triangles share, and
    // that is the common case rather than a corner case: a quad is two triangles
    // split along a diagonal, and a blocker's diagonal runs straight through the
    // middle of a light view aimed past it. On that line all three products are
    // zero plus float noise, and if both pieces round the wrong way the texel is
    // left empty -- so an opaque panel develops a one-texel-wide slit and the
    // light comes through it. Measured: 5.4% of an emitter leaking through a
    // blocker that covers it twice over, falling as 1/res, which is the
    // signature of a defect on a line rather than over an area.
    //
    // Letting the two pieces overlap by a hair is free, because both compute the
    // same depth from the same plane and the min resolves them identically. The
    // scale is the products' own magnitude, so it is dimensionless and needs no
    // tuning -- the same argument mbg_fill makes for its own eps.
    float tol = -1e-6 * (abs(b0) + abs(b1) + abs(b2) + 1e-30);
    if (b0 < tol || b1 < tol || b2 < tol) return MBG_EMPTY;

    float den = dot(pn, d);
    if (abs(den) < 1e-20) return MBG_EMPTY;
    float t = pc / den;
    if (t <= 0.0) return MBG_EMPTY;              // behind the receiver
    return mbg_pack_key(t * length(d), inv_far, ti);
}

// (visible, total) over this thread's texels, computing the visibility as it
// goes. Replaces clear + two rasterizations + lv_mass with one pass.
//
// The emitter is tested FIRST and the scene loop is skipped when it does not
// reach this texel, which the rasterized version could not do: it had to fill
// the whole scene into the buffer before it knew.
vec2 lv_mass_texel(uint res, vec3 N, uint emit_ti, uint tri_count, float inv_far,
                   uint lo, uint stride) {
    vec2 m = vec2(0.0);
    uint n = res * res;
    for (uint i = lo; i < n; i += stride) {
        vec3 d = lv_px_to_dir(vec2(float(i % res), float(i / res)) + vec2(0.5), res);
        float cosr = dot(N, d);
        if (cosr <= 0.0) continue;

        uint ke = lv_texel_key(emit_ti, d, inv_far);
        if (ke == MBG_EMPTY) continue;

        uint kv = MBG_EMPTY;
        for (uint t = 0u; t < tri_count; ++t) {
            uint k = lv_texel_key(t, d, inv_far);
            if (k < kv) kv = k;
        }

        float l2 = dot(d, d);
        float w = cosr / (l2 * l2);
        m.y += w;
        // One key step of tolerance, for an emitter coplanar with a surface.
        if (kv == MBG_EMPTY || (kv >> 16u) + 1u >= (ke >> 16u)) m.x += w;
    }
    return m;
}

// The sun's disc, same inversion. It needs only "is anything in the way", not
// which thing, so the scene loop stops at the first blocker.
vec2 lv_mass_disc_texel(uint res, vec3 N, float cos_r, uint tri_count, float inv_far,
                        uint lo, uint stride) {
    vec2 m = vec2(0.0);
    uint n = res * res;
    for (uint i = lo; i < n; i += stride) {
        vec3 d = lv_px_to_dir(vec2(float(i % res), float(i / res)) + vec2(0.5), res);
        float l = length(d);
        if (dot(d, lv_F) < cos_r * l) continue;
        float cosr = dot(N, d);
        if (cosr <= 0.0) continue;

        bool blocked = false;
        for (uint t = 0u; t < tri_count; ++t) {
            if (lv_texel_key(t, d, inv_far) != MBG_EMPTY) { blocked = true; break; }
        }

        float w = cosr / (l * l * l * l);
        m.y += w;
        if (!blocked) m.x += w;
    }
    return m;
}

// (visible, total) contribution mass over this thread's texels.
//
// Weighted by the receiver cosine and the projection's own solid angle: for a
// planar perspective grid dOmega = du dv / (1+u^2+v^2)^(3/2), and the cosine adds
// another 1/sqrt, so the weight is cos / (1+u^2+v^2)^2 with the constant du dv
// cancelling in the ratio. Texels below the receiver's horizon weigh nothing,
// which keeps an emitter straddling it consistent with the analytic magnitude,
// since that is clipped to the hemisphere too.
vec2 lv_mass(uint res, vec3 N, uint lo, uint hi, uint stride) {
    vec2 m = vec2(0.0);
    for (uint i = lo; i < hi; i += stride) {
        uint ke = s_lv_e[i];
        if (ke == MBG_EMPTY) continue;
        vec3 d = lv_px_to_dir(vec2(float(i % res), float(i / res)) + vec2(0.5), res);
        float cosr = dot(N, d);
        if (cosr <= 0.0) continue;
        float l2 = dot(d, d);
        float w = cosr / (l2 * l2);
        m.y += w;
        uint kv = s_lv[i];
        // One key step of tolerance, for an emitter coplanar with a surface --
        // Cornell's panel lies in the ceiling and atomicMin breaks that tie by
        // triangle index.
        if (kv == MBG_EMPTY || (kv >> 16u) + 1u >= (ke >> 16u)) m.x += w;
    }
    return m;
}

// (unlit, total) over the SUN's disc, weighted the same way lv_mass is.
//
// One buffer, not two. lv_mass divides by a rasterization of the emitter because
// the emitter is a polygon and only some of the frustum's texels are on it; a
// texel's quantization then cancels between the two passes. The sun has no
// polygon -- every texel inside the disc sees it, by definition of a light at
// infinity -- so the denominator is analytic and the numerator is simply the
// texels nothing was written into. Having no depth comparison at all is also why
// the sun needs no bias and cannot self-shadow numerically: lv_raster_tri's
// receiver-plane cull removes the receiver's own triangle before a key is ever
// packed, and after that, presence is the whole test.
//
// The disc test is what makes the frustum a CONE. The rasterized square
// circumscribes the disc, so a quarter of its texels are outside the sun and
// would otherwise be counted as sun that is never there.
vec2 lv_mass_disc(uint res, vec3 N, float cos_r, uint lo, uint hi, uint stride) {
    vec2 m = vec2(0.0);
    for (uint i = lo; i < hi; i += stride) {
        vec3 d = lv_px_to_dir(vec2(float(i % res), float(i / res)) + vec2(0.5), res);
        float l = length(d);
        if (dot(d, lv_F) < cos_r * l) continue;
        float cosr = dot(N, d);
        if (cosr <= 0.0) continue;
        float w = cosr / (l * l * l * l);
        m.y += w;
        if (s_lv[i] == MBG_EMPTY) m.x += w;
    }
    return m;
}

#endif
