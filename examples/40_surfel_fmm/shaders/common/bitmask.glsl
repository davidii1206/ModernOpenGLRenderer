#ifndef SGI_BITMASK_GLSL
#define SGI_BITMASK_GLSL

// ---------------------------------------------------------------------------
// Cone-bitmask visibility.
//
// An 8x8 grid of bits over the emitter's surface, one bit per stratified patch,
// packed into a uvec2. Every occluder between the receiver and the light is
// rasterized into the grid and OR-ed in; what is left unset is visible.
//
// WHY OR AND NOT A SUM. This example has been bitten twice by summed coverage:
// finding 14 measured the microbuffer reading a closed room 25.6% too open, and
// finding 19 had to add a calibration constant to stop a second layer drawing
// light out of that bias. The cause is that summing disc coverage double-counts
// overlapping occluders and then clamps. OR is idempotent, so overlapping
// occluders at different depths compose exactly -- this REPLACES the biased
// estimator rather than calibrating it. It is also the reason Bunnell-style
// multiplicative transmittance needs a multi-pass subtraction hack and this does
// not.
//
// No depth buffer and no z-test: unlike the microbuffer, everything between the
// receiver and the light occludes, full stop.
//
// uvec2 rather than uint64: GL_ARB_gpu_shader_int64 is an extension, while
// bitCount on uint is core GLSL 4.0 and this is a 4.6 context.
// ---------------------------------------------------------------------------

#include "micro.glsl"   // sgi_onb, for the degenerate cross-product case

// 16x16, not 8x8, and the reason is the penumbra rather than the umbra.
//
// The mask is the ONLY quantized thing in the direct term: everything else --
// the per-bit cosines, the areas, the distances -- is analytic. So a penumbra is
// reconstructed in steps of one bit, and at 8x8 one bit is 1/64 = 1.6% of the
// light. Measured down a column of the tall box's lit face, that came out as
// plateaus two to four pixels wide with ~1-unit jumps between them on a value of
// 36: a visible stair, and the last thing left in the direct term. At 16x16 the
// step is 0.4% and the stair falls below the 8-bit display quantum.
//
// 256 bits is four uvec2 rather than one. Held in registers and addressed by a
// four-way branch on the word index; GLSL will keep an indexed local array in
// scratch memory instead, which costs far more than the branch.
// Set this to 8u to go back to a 64-bit mask: the struct below holds 256 bits
// either way and every helper is written against kBitsN, so the resolution is a
// one-line change and nothing else has to move. It costs 2.8x on the direct pass
// (243 ms -> 674 ms at 512^2 on this machine), almost all of it in the occluder
// rasterization, whose bit-space footprint grows with the square of the edge.
const uint kBitsEdge = 16u;
const uint kBitsN    = kBitsEdge * kBitsEdge;

struct SgiMask { uvec2 w0, w1, w2, w3; };

SgiMask sgi_mask_zero() {
    SgiMask m; m.w0 = m.w1 = m.w2 = m.w3 = uvec2(0u); return m;
}

uvec2 sgi_bit(uint i) {
    return i < 32u ? uvec2(1u << i, 0u) : uvec2(0u, 1u << (i - 32u));
}

void sgi_mask_set(inout SgiMask m, uint b) {
    const uvec2 v = sgi_bit(b & 63u);
    const uint  k = b >> 6u;
    if      (k == 0u) m.w0 |= v;
    else if (k == 1u) m.w1 |= v;
    else if (k == 2u) m.w2 |= v;
    else              m.w3 |= v;
}

bool sgi_mask_test(SgiMask m, uint b) {
    const uvec2 v = sgi_bit(b & 63u);
    const uint  k = b >> 6u;
    const uvec2 w = k == 0u ? m.w0 : (k == 1u ? m.w1 : (k == 2u ? m.w2 : m.w3));
    return (w & v) != uvec2(0u);
}

void sgi_mask_merge(inout SgiMask a, SgiMask b) {
    a.w0 |= b.w0; a.w1 |= b.w1; a.w2 |= b.w2; a.w3 |= b.w3;
}

uint sgi_popcount(SgiMask m) {
    return uint(bitCount(m.w0.x) + bitCount(m.w0.y) + bitCount(m.w1.x) + bitCount(m.w1.y) +
                bitCount(m.w2.x) + bitCount(m.w2.y) + bitCount(m.w3.x) + bitCount(m.w3.y));
}

// Against kBitsN rather than against all four words, so this stays correct when
// kBitsEdge is 8 and the upper words are structurally zero.
bool sgi_mask_full(SgiMask m) { return sgi_popcount(m) == kBitsN; }

// The emitter rectangle. Corners are centre +/- half_u +/- half_v; mirrors
// GpuEmitter in direct.hpp.
struct SgiEmitter {
    vec3  centre; float area;
    vec3  half_u; float rad_r;
    vec3  half_v; float rad_g;
    vec3  normal; float rad_b;
};

SgiEmitter sgi_emitter(vec4 p0, vec4 p1, vec4 p2, vec4 p3) {
    SgiEmitter e;
    e.centre = p0.xyz; e.area  = p0.w;
    e.half_u = p1.xyz; e.rad_r = p1.w;
    e.half_v = p2.xyz; e.rad_g = p2.w;
    e.normal = p3.xyz; e.rad_b = p3.w;
    return e;
}

// Centre of bit (i, j) in the rectangle's normalized [-1,1]^2 coordinates. A
// regular 8x8 stratification of a RECTANGLE is equal-area by construction, so a
// plain popcount is the correct estimator and the Shirley-Chiu concentric map is
// not needed -- that map exists to make a square-to-DISK mapping area-preserving,
// which is the case to reach for when a disk proxy exists.
vec2 sgi_bit_uv(uint i, uint j) {
    return (vec2(float(i), float(j)) + 0.5) * (2.0 / float(kBitsEdge)) - 1.0;
}

// An occluder's shadow on the emitter plane: the ellipse
//
//     c + a0*cos(t) + a1*sin(t),   t in [0, 2pi)
//
// in the rectangle's normalized [-1,1]^2 coordinates. a0 and a1 are conjugate
// semi-diameters, not necessarily the principal axes and not necessarily
// orthogonal -- which is exactly why they are stored as vectors. The inside test
// only needs M = [a0 a1] to be invertible, and the principal axes are never
// needed.
struct SgiFootprint {
    vec2 c;
    vec2 a0;
    vec2 a1;
};

// Where the ray P -> X meets the emitter plane, in normalized rect coordinates.
// `ilu2`/`ilv2` are 1/|half_u|^2 and 1/|half_v|^2, hoisted by the caller.
bool sgi_plane_uv(vec3 P, vec3 X, SgiEmitter e, float ilu2, float ilv2, out vec2 uv) {
    const vec3  d   = X - P;
    const float den = dot(d, e.normal);
    if (abs(den) < 1e-9) return false;
    const float t = dot(e.centre - P, e.normal) / den;
    if (t <= 0.0) return false;
    const vec3 h = P + d * t - e.centre;
    uv = vec2(dot(h, e.half_u) * ilu2, dot(h, e.half_v) * ilv2);
    return true;
}

// Project one occluder disc onto the emitter plane as seen from P.
//
// THE ORIENTATION IS PART OF THE ANSWER. The first version of this returned two
// scalar semi-axes and assigned the unforeshortened one to `half_u` and the
// foreshortened one to `half_v`, whatever the disc's actual orientation was.
// That is wrong by construction: rotate the occluder 90 degrees about the line
// of sight and its shadow is unchanged, but the mask it wrote flipped between
// over- and under-occluding. It leaked along every edge whose surfels happened
// to foreshorten along u.
//
// The silhouette of a disc seen from P is an ellipse whose unforeshortened axis
// lies along cross(n_o, w) and whose foreshortened axis is perpendicular to
// both. Both axes are perpendicular to w, so both are carried through the
// projection exactly by the same construction as the centre.
//
// THICKNESS, NOT A CLAMP. An edge-on disc projects to a line and occludes
// nothing, so a surface tiled with discs goes transparent when the light grazes
// it. The old fix clamped the minor axis to a fixed fraction of the major one,
// which is a constant over-occlusion applied hardest exactly where it is least
// justified -- a ceiling disc seen edge-on from a wall point a few centimetres
// below the ceiling was inflated to 0.75 of its full radius and blacked out the
// light, which is the ragged dark band along every concave edge.
//
// A surfel is better modelled as a thin oblate SPHEROID of semi-axes
// (r, r, thick*r). Its silhouette semi-minor axis is
//
//     r * (|cos| + thick * |sin|)
//
// which is exact for that solid, equals r face-on, and degrades to thick*r
// edge-on rather than to a constant. The difference is that a LONG grazing path
// through a wall crosses many discs and their slivers still OR together into an
// opaque mask, while a SHORT one near a corner does not -- which is the correct
// behaviour in both cases, and the reason this is a model rather than a fudge.
bool sgi_project_occluder(vec3 P, vec3 C, vec3 n_o, float r_o,
                          SgiEmitter e, float thick, out SgiFootprint fp)
{
    const vec3  v    = C - P;
    const float dist = length(v);
    if (dist < 1e-6) return false;
    const vec3 w = v / dist;

    const float lu2 = dot(e.half_u, e.half_u);
    const float lv2 = dot(e.half_v, e.half_v);
    if (lu2 < 1e-12 || lv2 < 1e-12) return false;
    const float ilu2 = 1.0 / lu2, ilv2 = 1.0 / lv2;

    // Where the ray P -> C meets the emitter's plane.
    const float num   = dot(e.centre - P, e.normal);
    const float denom = dot(v, e.normal);
    if (abs(denom) < 1e-9) return false;
    const float s0 = num / denom;                       // hit = P + s0 * v
    if (s0 <= 0.0 || s0 <= 1.0) return false;           // behind P, or past the light

    const vec3 hit = P + v * s0;
    const vec3 d   = hit - e.centre;
    fp.c = vec2(dot(d, e.half_u) * ilu2, dot(d, e.half_v) * ilv2);

    // Silhouette axes, both perpendicular to the line of sight.
    vec3 maj = cross(n_o, w);
    const float lm = length(maj);
    if (lm > 1e-6) maj /= lm;
    else { vec3 tb; sgi_onb(w, maj, tb); }              // disc face-on: any axis
    const vec3 mnr = cross(w, maj);                     // unit: maj and w are unit and orthogonal

    const float ct = abs(dot(n_o, w));
    const float st = sqrt(max(1.0 - ct * ct, 0.0));
    const vec3  E0 = maj * r_o;
    const vec3  E1 = mnr * (r_o * (ct + thick * st));

    // Carry an offset E perpendicular to w through the same projection. The ray
    // P -> (C + E) meets the plane at P + s*(v + E), so the shadow's offset is
    // (s - s0)*v + s*E -- exact, and only one divide more than the paraxial
    // approximation (s0*E), which is wrong by a factor that grows with how
    // obliquely the shadow lands on the light.
    vec2 duv[2];
    for (int k = 0; k < 2; ++k) {
        const vec3  E   = k == 0 ? E0 : E1;
        const float den = denom + dot(E, e.normal);
        const float s   = abs(den) > 1e-9 ? num / den : s0;
        const vec3  off = (s > 0.0) ? ((s - s0) * v + s * E) : (s0 * E);
        duv[k] = vec2(dot(off, e.half_u) * ilu2, dot(off, e.half_v) * ilv2);
    }
    fp.a0 = duv[0];
    fp.a1 = duv[1];
    return true;
}

// --- edge cut planes -------------------------------------------------------
//
// The ellipse above is the silhouette of a WHOLE disc, and a whole disc is the
// wrong shape at a mesh edge: the union of discs whose centres lie inside a
// surface extends up to r past its boundary, so every silhouette is dilated by
// one surfel radius and a straight edge made of a row of circles comes out
// scalloped at the surfel spacing. Inflating the radius to seal the interior
// (u_occ) makes that worse, which is why one radius cannot fix both.
//
// So clip. Each surfel carries up to kSurfelCuts planes, packed as
// (n, -dot(anchor, n)) and derived at bake time from the mesh's own sharp and
// boundary edges, so they land exactly on the triangle borders whatever the
// sampling did. Ported from example 38, where the same planes make mirror
// reflections of a disc cloud terminate cleanly at creases.
//
// Applied PER BIT and only to bits the ellipse already accepted: the bit is a
// known point on the light, so the ray from the receiver to it meets the
// occluder's tangent plane at one point, and that point is either inside the
// clipped disc or not. Nothing is approximated here -- this is the exact test,
// with the ellipse serving only as the cheap bound that says which bits to try.
//
// The same per-bit ray also answers a question the ellipse CANNOT express, and
// getting it wrong was a bigger error than the dilation: whether the occluder is
// actually BETWEEN the receiver and that point on the light. See
// sgi_slab_blocks().
const int kSurfelCuts = 4;

layout(std430, binding = 22) readonly buffer SurfCut { vec4 s_cut[]; };

bool sgi_has_cuts(uint s) { return dot(s_cut[s * uint(kSurfelCuts)].xyz,
                                       s_cut[s * uint(kSurfelCuts)].xyz) > 0.0; }

bool sgi_cut_rejects(uint s, vec3 p) {
    const uint base = s * uint(kSurfelCuts);
    for (int k = 0; k < kSurfelCuts; ++k) {
        const vec4 pl = s_cut[base + uint(k)];
        // An empty slot is the zero plane: dot(p, 0) + 0 == 0, never rejects.
        if (dot(p, pl.xyz) + pl.w > 0.0) return true;
    }
    return false;
}

// IS THE OCCLUDER ACTUALLY BETWEEN THE RECEIVER AND THE LIGHT?
//
// The projection answers "does the occluder's silhouette cover this direction",
// which is not the same question, and the difference is not academic. A wall
// disc sitting a centimetre above a floor receiver at the wall/floor junction is
// seen EDGE-ON, so its silhouette is a sliver -- but its centre is only ~r away,
// so the projection magnifies that sliver by d_emitter/r (about 120x here) and it
// blankets the whole panel. The floor in front of the back wall then reads as
// shadowed, in a row of teeth at the surfel spacing. Nothing physical is
// happening: every ray from that receiver to the light goes FORWARD, away from
// the wall, and never crosses the wall's plane at all.
//
// `sgi_same_surface` cannot catch this -- the two normals are perpendicular, so
// they are emphatically not the same surface -- and no thickness or radius
// setting touches it, which is why it survived every parameter sweep.
//
// The occluder is the oblate spheroid of finding 23, so the exact test is a slab
// crossing: the ray P + s*d, s in (0,1), must enter |dot(q - C, n_o)| <= slab_h.
// Solving both plane crossings gives an s-interval, and the occluder blocks iff
// that interval meets (0, 1). The near-parallel case falls out as a limit rather
// than needing its own branch: |dn| tiny leaves an interval that is either the
// whole line (the receiver is inside the slab, so the disc really does block) or
// empty (it is outside, so it cannot).
bool sgi_slab_blocks(float cnum, float dn, float slab_h) {
    const float lo = cnum - slab_h, hi = cnum + slab_h;
    if (abs(dn) < 1e-12) return lo <= 0.0 && hi >= 0.0;
    const float a = lo / dn, b = hi / dn;
    return max(min(a, b), 0.0) < min(max(a, b), 1.0);
}

// Rasterize an oriented ellipse in normalized rect coordinates into the mask.
//
// `cut` is the occluder's index, or 0xFFFFFFFF for no clipping. P/C/n_o/slab_h/e
// reconstruct the ray for each accepted bit; `slab_h` is the spheroid's slab
// half-thickness, thick * r_effective.
// Accumulates INTO the caller's running mask rather than returning its own, so
// that a bit some earlier occluder already set can be skipped instead of
// re-derived. OR is idempotent -- setting a set bit is a no-op -- so this is
// bit-identical to accumulating separately and merging, and it is what makes an
// umbra cheap: once the mask fills, every later occluder does almost no work.
void sgi_raster_ellipse_cut(inout SgiMask m, SgiFootprint fp, uint cut,
                            vec3 P, vec3 C, vec3 n_o, float slab_h, SgiEmitter e) {

    // Axis-aligned bound of the ellipse: |a0| + |a1| per component is the
    // support of c + a0 cos t + a1 sin t, componentwise and conservatively.
    const vec2 ext = abs(fp.a0) + abs(fp.a1);
    if (fp.c.x - ext.x >  1.0 || fp.c.x + ext.x < -1.0 ||
        fp.c.y - ext.y >  1.0 || fp.c.y + ext.y < -1.0) return;

    const float step = 2.0 / float(kBitsEdge);
    const int lo_i = max(int(floor((fp.c.x - ext.x + 1.0) / step)), 0);
    const int hi_i = min(int(ceil ((fp.c.x + ext.x + 1.0) / step)), int(kBitsEdge) - 1);
    const int lo_j = max(int(floor((fp.c.y - ext.y + 1.0) / step)), 0);
    const int hi_j = min(int(ceil ((fp.c.y + ext.y + 1.0) / step)), int(kBitsEdge) - 1);

    // Inside test: solve [a0 a1] s = p - c and check |s| <= 1. A degenerate
    // matrix means the shadow collapsed to a segment in rect space, which
    // happens only when the receiver lies almost in the emitter's own plane --
    // fall back to the bounding box there rather than dropping the occluder,
    // because dropping it is a leak and the box is at most one bit wide.
    const float det = fp.a0.x * fp.a1.y - fp.a0.y * fp.a1.x;
    const bool  ok  = abs(det) > 1e-14;
    const float inv = ok ? 1.0 / det : 0.0;

    // Clipping is per surfel; the slab test is unconditional.
    const bool  clip = cut != 0xFFFFFFFFu && sgi_has_cuts(cut);
    const float cnum = dot(C - P, n_o);

    for (int j = lo_j; j <= hi_j; ++j) {
        for (int i = lo_i; i <= hi_i; ++i) {
            const uint bit = uint(j) * kBitsEdge + uint(i);
            if (sgi_mask_test(m, bit)) continue;      // already shadowed
            const vec2 uv = sgi_bit_uv(uint(i), uint(j));
            const vec2 q  = uv - fp.c;
            if (ok) {
                const vec2 s = vec2( fp.a1.y * q.x - fp.a1.x * q.y,
                                    -fp.a0.y * q.x + fp.a0.x * q.y) * inv;
                if (dot(s, s) > 1.0) continue;
            }
            const vec3  X  = e.centre + e.half_u * uv.x + e.half_v * uv.y;
            const vec3  d  = X - P;
            const float dn = dot(d, n_o);
            if (!sgi_slab_blocks(cnum, dn, slab_h)) continue;
            // Clip at the mesh edge, at the point where this ray actually meets
            // the occluder's disc.
            //
            // Head-on that is the intersection with the occluder's plane. Edge-on
            // it is not: `cnum / dn` runs away to infinity as dn -> 0, so that
            // branch used to be SKIPPED within ~6 degrees of edge-on, on the
            // argument that an unclipped bit over-occludes slightly while a
            // wrongly dropped one leaks.
            //
            // Slightly, except in the one configuration where edge-on is the
            // rule rather than the exception: a receiver on the floor beside a
            // wall, looking up at a light overhead. Every one of those rays runs
            // nearly parallel to the wall, so no wall surfel near the junction
            // was ever clipped at the junction -- and at u_occ = 2.0 their discs
            // reach two radii out across the floor. That is the floor's dark
            // fringe along every concave corner.
            //
            // Near edge-on the well-conditioned point is instead the one on the
            // ray closest to the disc's centre, projected into the disc's plane.
            // Both branches ask the same question and agree where they overlap;
            // the second is simply the one that survives dn -> 0.
            if (clip) {
                vec3 q;
                if (abs(dn) > 0.1 * length(d)) {
                    q = P + d * (cnum / dn);
                } else {
                    const float sc = clamp(dot(C - P, d) / max(dot(d, d), 1e-20),
                                           0.0, 1.0);
                    const vec3  rp = P + d * sc;
                    q = rp - n_o * dot(rp - C, n_o);
                }
                if (sgi_cut_rejects(cut, q)) continue;
            }
            sgi_mask_set(m, bit);
        }
    }
}

#endif
