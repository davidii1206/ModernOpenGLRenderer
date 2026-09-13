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
bool lv_setup(vec3 P, vec3 N, MbgTri tr) {
    lv_P = P;
    lv_N = N;
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
void lv_setup_dir(vec3 P, vec3 N, vec3 dir, float tan_r, float near_d) {
    lv_P = P;
    lv_N = N;
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
            if (t <= 0.0) continue;
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
