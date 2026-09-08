#ifndef SGI_NEE_GLSL
#define SGI_NEE_GLSL

// ---------------------------------------------------------------------------
// Next event estimation: the direct term, shared by the per-surfel pass
// (nee_direct.comp) and the per-pixel pass (nee_pixel.comp).
//
// The two passes differ ONLY in what a receiver is. Everything below -- the
// frustum walk, the occluder splat, the per-bit quadrature -- is identical, and
// it is shared textually rather than duplicated because the two must agree
// exactly: the per-surfel pass feeds the bounce transport and the per-pixel pass
// feeds the image, and a divergence between them would show up as a mismatch
// between direct and indirect that is impossible to attribute.
//
// PER-BIT ANALYTIC CONTRIBUTION, not form factor times a scalar visibility.
// Each bit is a patch of the light of known area, direction and cosines, so the
// visible bits are summed directly:
//
//     E = SUM over visible bits  L_e * cos_r * cos_e * dA / d^2
//
// That is a 64-point deterministic quadrature of the area integral. It matters
// here rather than being pedantry: Cornell's panel subtends 25 degrees from a
// floor point half a unit below it, and at that width visibility CORRELATES with
// direction across the light -- the near edge and the far edge of the panel are
// occluded differently and carry different cosines. Multiplying one form factor
// by a scalar visibility fraction throws that correlation away.
//
// It also means no separate polygon form factor is needed, and no weight table
// for a radial light profile: the per-bit cosines already carry it.
// ---------------------------------------------------------------------------

#include "surfel.glsl"
#include "bitmask.glsl"

layout(std430, binding = 8)  readonly buffer CellSC   { uvec2 cell_sc[]; };
layout(std430, binding = 9)  readonly buffer CellIt   { uint  cell_item[]; };
layout(std430, binding = 10) readonly buffer CellPR   { vec4  cell_pr[]; };
layout(std430, binding = 19) readonly buffer Emitters { vec4  emitters[]; };
layout(std430, binding = 21) readonly buffer Macro    { uint  macro_bits[]; };

uniform uint  u_emitters;
uniform vec3  u_grid_min;
uniform ivec3 u_grid_res;
uniform ivec3 u_macro_res;
uniform float u_inv_cell;
uniform float u_cell;
uniform float u_thick;      // surfel slab half-thickness, in radii; see bitmask.glsl
// THE OCCLUDER RADIUS IS NOT THE ENERGY RADIUS. The bake sizes surfels so that
// sum(pi r^2) == A exactly (gate 6), which is the right condition for a disc to
// carry the right amount of energy -- and the wrong one for it to SEAL. Equal
// circles cannot tile a plane: the densest possible packing covers 0.9069 of it,
// and a Vogel spiral is looser still, so a surface tiled at coverage 1.0 has real
// holes in it and a shadow ray finds them. Gate 12 measures 17% of the light
// straight through a head-on wall at coverage 1.0, at every thickness -- thickness
// cannot help, because a head-on disc is already at its full radius.
//
// So scale the radius used for VISIBILITY only. Energy, form factors and the
// microbuffer keep the bake's radius, so nothing downstream of gate 2 or gate 6
// moves; the shadow query gets discs that overlap enough to close. Coverage goes
// as the square, so the shipped u_occ = 2.0 is coverage 4.
//
// HOW MUCH IS ENOUGH DEPENDS ON THE SAMPLER, not on the coverage number alone --
// see gate 12. A Vogel spiral seals at 1.25; the bake's per-triangle R2 sequence,
// which is what the real scene has, still leaks at 1.5 along grazing paths near a
// clipped boundary and needs 2.0. Inflating this far is only affordable because
// the cut planes stop it from dilating silhouettes (finding 27); before those,
// 2.0 cost real accuracy in the penumbra.
uniform float u_occ;        // occluder radius scale, for visibility only
uniform uint  u_cuts;       // 1 = clip occluder discs at the mesh's feature edges
uniform float u_bias;       // receiver offset along its own normal, in radii
uniform float u_self_cos;   // an occluder this parallel to the receiver ...
uniform float u_self_tol;   // ... and this close to its plane is the SAME SURFACE

// One bit per 4x4x4 block of cells.
//
// THE SHAPE OF THE WALK MATTERS AND THE FIRST VERSION GOT IT WRONG. Marching the
// axis and scanning a fixed box neighbourhood around each step has to cap the
// box or the cost explodes, and with a WIDE light the cap silently truncates the
// query: Cornell's panel is 14.6 cells across at its own plane, a cap of 6 left
// 59% of the frustum unscanned, and the umbra came out 9.6x too bright. A cone
// half-angle of 25 degrees is not the sun.
//
// So enumerate the frustum's cells once, through a coarse occupancy test.
// Cornell's interior is hollow, so nearly every block in the open volume between
// the floor and the ceiling is rejected on a single bit. What survives is the
// surfaces, which is what can actually occlude.
bool sgi_macro_occupied(ivec3 m) {
    if (any(lessThan(m, ivec3(0))) || any(greaterThanEqual(m, u_macro_res))) return false;
    const uint mb = uint(m.x) + uint(u_macro_res.x) *
                    (uint(m.y) + uint(u_macro_res.y) * uint(m.z));
    return (macro_bits[mb >> 5u] & (1u << (mb & 31u))) != 0u;
}

// Conservative sphere-vs-frustum: is any part of a sphere of radius R at C
// inside the cone that opens from P to the emitter rectangle?
bool sgi_in_frustum(vec3 C, float R, vec3 P, vec3 dir, float d_e, float half_w) {
    const vec3  v = C - P;
    const float t = dot(v, dir);
    if (t < -R || t > d_e + R) return false;
    const float dperp = length(v - dir * t);
    const float rad = half_w * clamp(t / max(d_e, 1e-6), 0.0, 1.0);
    return dperp <= rad + R;
}

// A FLAT SURFACE DOES NOT SHADOW ITSELF, and the receiver's own neighbours are
// the occluders closest to it, so they are the ones a projection error hurts
// most: they are seen edge-on, which is precisely the case the thickness model
// has to inflate, and they surround the receiver on every side.
//
// Rejecting them explicitly is what lets the receiver bias come down. The bias
// existed to lift the receiver out of its own plane so those neighbours stopped
// registering -- but lifting a floor point a full surfel radius also lifts it
// out from behind whatever is standing on the floor, which is a leak at every
// contact and every concave edge. Test for "same surface" directly instead:
// near-parallel normals AND near-zero offset from the receiver's tangent plane.
// A crease fails the first test and a step fails the second, so both still
// occlude.
bool sgi_same_surface(vec3 p_surf, vec3 nP, vec3 C, vec3 n_o, float r_o) {
    if (dot(n_o, nP) < u_self_cos) return false;
    return abs(dot(C - p_surf, nP)) < u_self_tol * r_o;
}

uint g_cells; uint g_cand; uint g_proj;
SgiMask sgi_trace_mask(vec3 P, vec3 p_surf, vec3 nP, SgiEmitter e, uint self) {
    SgiMask mask = sgi_mask_zero();

    const vec3  to_e = e.centre - P;
    const float d_e  = length(to_e);
    if (d_e < 1e-6) return mask;
    const vec3  dir  = to_e / d_e;
    const float half_w = length(e.half_u) + length(e.half_v);

    // Bounding box of the frustum, in macro blocks.
    const vec3 far_c = P + dir * d_e;
    const vec3 lo = min(P, far_c - vec3(half_w)) - vec3(u_cell);
    const vec3 hi = max(P, far_c + vec3(half_w)) + vec3(u_cell);
    const ivec3 mlo = clamp(ivec3(floor((lo - u_grid_min) * u_inv_cell)) / 4,
                            ivec3(0), u_macro_res - 1);
    const ivec3 mhi = clamp(ivec3(floor((hi - u_grid_min) * u_inv_cell)) / 4,
                            ivec3(0), u_macro_res - 1);

    const float mstep = u_cell * 4.0;
    const float mrad  = mstep * 0.8660254;   // half-diagonal of a macro block
    const float crad  = u_cell * 0.8660254;

    for (int mz = mlo.z; mz <= mhi.z; ++mz)
    for (int my = mlo.y; my <= mhi.y; ++my)
    for (int mx = mlo.x; mx <= mhi.x; ++mx) {
        const ivec3 m = ivec3(mx, my, mz);
        if (!sgi_macro_occupied(m)) continue;
        const vec3 mc = u_grid_min + (vec3(m) * 4.0 + 2.0) * u_cell;
        if (!sgi_in_frustum(mc, mrad, P, dir, d_e, half_w)) continue;

        for (int cz = 0; cz < 4; ++cz)
        for (int cy = 0; cy < 4; ++cy)
        for (int cx = 0; cx < 4; ++cx) {
            const ivec3 g = m * 4 + ivec3(cx, cy, cz);
            if (any(greaterThanEqual(g, u_grid_res))) continue;
            const vec3 gc = u_grid_min + (vec3(g) + 0.5) * u_cell;
            if (!sgi_in_frustum(gc, crad, P, dir, d_e, half_w)) continue;

            const uint ci = uint(g.x) + uint(u_grid_res.x) *
                            (uint(g.y) + uint(u_grid_res.y) * uint(g.z));
            g_cells += 1u;
            const uvec2 sc = cell_sc[ci];
            for (uint k = 0u; k < sc.y; ++k) {
                const uint idx = sc.x + k;
                g_cand += 1u;
                const uint j   = cell_item[idx];
                if (j == self) continue;
                if (surfel_is_emissive(j)) continue;   // the light is not its own occluder
                const vec4 pr = cell_pr[idx];
                const vec3 nj = surfel_normal(j);
                if (sgi_same_surface(p_surf, nP, pr.xyz, nj, pr.w)) continue;
                SgiFootprint fp;
                if (!sgi_project_occluder(P, pr.xyz, nj, pr.w * u_occ, e, u_thick, fp)) continue;
                // Clip against this surfel's own cut planes. Inflating by u_occ
                // above and clipping here is the pair that separates the two
                // opposite errors: interior discs must overlap to seal, boundary
                // discs must not overrun the geometry they represent.
                g_proj += 1u;
                sgi_raster_ellipse_cut(mask, fp, u_cuts != 0u ? j : 0xFFFFFFFFu,
                                       P, pr.xyz, nj,
                                       u_thick * pr.w * u_occ, e);
            }
        }
        if (sgi_mask_full(mask)) return mask;   // fully shadowed; nothing left to find
    }
    return mask;
}

// The direct irradiance at a receiver, and the fraction of the light's ENERGY it
// can see. `radius` is the receiver's own surfel radius (a pixel borrows the
// set's global one) and only sets the bias and the self-surface tolerance.
void sgi_nee_direct(vec3 p_surf, vec3 nP, float radius, uint self,
                    out vec3 E, out float vis)
{
    g_cells = 0u; g_cand = 0u; g_proj = 0u;
    const vec3 p = p_surf + nP * (u_bias * radius);

    vec3  sum = vec3(0.0);
    float vis_acc = 0.0, vis_w = 0.0;

    for (uint ei = 0u; ei < u_emitters; ++ei) {
        const SgiEmitter e = sgi_emitter(emitters[ei * 4u + 0u], emitters[ei * 4u + 1u],
                                         emitters[ei * 4u + 2u], emitters[ei * 4u + 3u]);
        // Receiver must face the light and the light must face the receiver.
        const vec3 to_c = e.centre - p;
        if (dot(to_c, nP) <= 0.0) continue;
        if (dot(-to_c, e.normal) <= 0.0) continue;

        const SgiMask mask = sgi_trace_mask(p, p_surf, nP, e, self);
        const vec3  Le   = vec3(e.rad_r, e.rad_g, e.rad_b);
        const float dA   = e.area / float(kBitsN);

        vec3 acc = vec3(0.0);
        float unocc = 0.0, seen = 0.0;
        for (uint b = 0u; b < kBitsN; ++b) {
            const vec2 uv = sgi_bit_uv(b % kBitsEdge, b / kBitsEdge);
            const vec3 X  = e.centre + e.half_u * uv.x + e.half_v * uv.y;
            const vec3 d  = X - p;
            const float d2 = dot(d, d);
            if (d2 < 1e-12) continue;
            const float inv = inversesqrt(d2);
            const vec3  w   = d * inv;
            const float cr  = dot(nP, w);
            const float ce  = dot(e.normal, -w);
            if (cr <= 0.0 || ce <= 0.0) continue;

            const float term = cr * ce * dA / d2;
            unocc += term;                                  // for the visibility ratio
            if (sgi_mask_test(mask, b)) continue;            // occluded
            acc  += Le * term;
            seen += term;
        }
        sum += acc;
        // Energy-weighted visibility, so the scalar reported downstream reflects
        // how much LIGHT is visible rather than how many bits are clear.
        vis_acc += seen;
        vis_w   += unocc;
    }

    E = sum;
    // Noise-free by construction: 256 stratified samples of the emitter with no
    // random sampling anywhere. This replaces the microbuffer's emissive-coverage
    // by-product as the edge-stop key (finding 19), which was built from ~4
    // buckets and carried the microbuffer's own sampling noise.
    vis = vis_w > 1e-12 ? vis_acc / vis_w : 0.0;
}

#endif
