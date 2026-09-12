#ifndef MBG_EMITTER_GLSL
#define MBG_EMITTER_GLSL

// ---------------------------------------------------------------------------
// The direct term's shared parts: everything a receiver needs regardless of how
// its visibility is found.
//
// Design doc section 3 keeps direct lighting out of the GI pass entirely. This
// example does that, and then splits it again by RECEIVER, because the two have
// completely different resolution requirements:
//
//   raster.comp        one receiver per secondary camera. Feeds the bounce
//                      transport, where a soft answer is fine.
//   direct_pixel.comp  one receiver per PIXEL. Feeds the image, where it is the
//                      sharpest term there is.
//
// Both get their VISIBILITY the same way and the only way anything in this
// renderer does: from the rasterized hemisphere's depth sort, via
// mbg_emitter_visible_fraction() below. Both call the magnitude function below
// that, so the two paths cannot drift apart.
// The triangle buffer and the emitter list are declared by the including shader;
// only the maths lives here.
// ---------------------------------------------------------------------------

#include "hemi.glsl"

vec3 mbg_emitter_unshadowed(vec3 pos, vec3 nrm, MbgTri tr, bool two_sided,
                            float plane_eps, float emissive_scale) {
    vec3 v[MBG_CLIP_MAX];
    v[0] = tr.p0.xyz - pos;
    v[1] = tr.p1.xyz - pos;
    v[2] = tr.p2.xyz - pos;

    // Which face is turned toward the receiver, and how far off its plane the
    // receiver is. h > 0 means in front of the emitting face.
    float h = -dot(tr.n.xyz, v[0]);
    if (h < 0.0 && tr.n.w == 0.0 && !two_sided) return vec3(0.0);

    // COPLANAR RECEIVERS CONTRIBUTE NOTHING, AND THE FORMULA DOES NOT KNOW THAT.
    //
    // An emitter whose plane contains the receiver is edge-on: measure zero, no
    // irradiance. But on the sphere that polygon degenerates to a great circle,
    // and if the receiver's projection falls INSIDE the triangle the three edge
    // terms all take the same sign and the sum comes out at 2*PI -- so the
    // formula reports a full hemisphere of the emitter's radiance instead of
    // nothing.
    //
    // It is not a rounding error, it is the worst possible answer, and it lands
    // exactly where it hurts most: every hit point on the panel itself, and
    // every child camera a tile places there. At the panel's emitted radiance of
    // 17 that single term dominates its whole tile, which is where the panel-
    // shaped bright blobs in this example's earlier images came from. The
    // `series` gate caught it as a 29% energy overshoot at two levels.
    //
    // The threshold is the camera bias, because that is exactly how far off its
    // surface this example places a receiver: anything closer than that is on
    // the emitter, not near it. A genuinely coplanar surface -- Cornell's
    // ceiling, which the panel is flush with -- receives nothing from it, which
    // is the physically correct answer and not an approximation.
    //
    // BOTH CONDITIONS ARE REQUIRED. Rejecting on the height alone also throws
    // away emitters that merely PASS THROUGH the receiver while standing
    // perpendicular to it -- a wall meeting a floor, or any hit point that lands
    // on a box edge -- and those cover half the receiver's hemisphere, not none
    // of it. The texeldir gate caught exactly that: hit points landing on the
    // gate box's edges read half their true irradiance, and because a symmetric
    // box puts many texel centres precisely on its own edges, that was 12% of
    // the whole hemisphere at an 8x8 target.
    if (abs(h) < plane_eps && abs(dot(tr.n.xyz, nrm)) > 0.9) return vec3(0.0);

    vec3 c[MBG_CLIP_MAX];
    int n = mbg_clip_plane(v, 3, nrm, c);
    if (n < 3) return vec3(0.0);

    float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        vec3 a = normalize(c[i]);
        vec3 b = normalize(c[i + 1 == n ? 0 : i + 1]);
        vec3 x = cross(a, b);
        float len = length(x);
        if (len < 1e-9) continue;
        sum += acos(clamp(dot(a, b), -1.0, 1.0)) * dot(nrm, x / len);
    }
    return tr.emission.xyz * emissive_scale * (0.5 * abs(sum));
}


// The fraction of an emitter that a receiver can actually see, entirely from the
// depth sort.
//
// The numerator is the cosine-weighted mass of the texels this emitter WON in
// the full buffer; the denominator is the mass it covers with nothing else in
// the scene, from the emitters-only buffer. Both come off the same texel grid,
// so the grid's quantization -- the thing that made a raw quadrature estimate of
// the direct term useless -- cancels exactly, and what is left is a proper
// fraction in [0,1] that is 1 in the open, 0 in umbra and a ramp across a
// penumbra.
//
// NOTHING HERE COMPARES A DEPTH AGAINST ANYTHING. The atomicMin already did
// that, per texel, along the texel's own direction, which is the only place the
// comparison is exact. An earlier version tested emitter samples against the
// stored depth and paid for it twice: first with false self-occlusion wherever a
// surface was grazing (a diagonal cross-hatch over the whole image), then with a
// ray cast to paper over it. The depth sort takes care of it.
//
// Call it with the texel range this thread owns and reduce across the workgroup;
// `x` is the numerator, `y` the denominator.
vec2 mbg_emitter_mass(uint tri_index, uint lo, uint hi, uint stride) {
    vec2 m = vec2(0.0);
    for (uint i = lo; i < hi; i += stride) {
        uint ke = s_emit[i];
        if (ke == MBG_EMPTY || mbg_key_tri(ke) != tri_index) continue;
        m.y += quad[i].w;                       // the emitter reaches this texel

        // ...and nothing got in front of it. Both depths were produced by the
        // same rasterizer along the SAME texel direction, so this comparison is
        // exact -- it is the depth sort's own answer read back, not a shadow-map
        // lookup with a slope to worry about.
        //
        // The +1 of tolerance is one step of the 16-bit key, and it is what makes
        // a coplanar emitter work at all: Cornell's panel lies IN the ceiling, so
        // the two are at identical depth in every texel the panel covers and
        // atomicMin breaks the tie by triangle index. Without the tolerance the
        // ceiling wins half of them, the visible fraction comes out below 1 in
        // full view, and the error moves from pixel to pixel -- 4x the
        // reference's roughness on the back wall, measured.
        uint kv = s_vis[i];
        if (kv == MBG_EMPTY || (kv >> 16u) + 1u >= (ke >> 16u)) m.x += quad[i].w;
    }
    return m;
}

// An emitter smaller than one texel wins no texel centre, and then there is no
// ratio to take: the denominator is zero. This is the resolution floor of the
// method, and what happens at it matters more than it sounds -- at a 32x32
// target a surprising number of Cornell's receivers see the panel across fewer
// than one texel, and answering "not visible" there put dark speckle along every
// surface junction in the image (2.64x the reference's roughness, against 1.37x
// when the same case answered "visible").
//
// So the degenerate case degrades to a single depth comparison, still the depth
// sort's own numbers and still no ray: look up the texel the emitter's centroid
// falls in, and ask whether whatever won it is behind the emitter's own plane
// along that same texel's direction. Sub-texel emitters get a one-texel-wide
// shadow boundary, which is the honest answer at that resolution.
float mbg_emitter_fallback(MbgTri tr, uint res, float inv_far) {
    vec3 cl = mbg_to_local((tr.p0.xyz + tr.p1.xyz + tr.p2.xyz) / 3.0 - g_P);
    if (cl.z <= 0.0) return 0.0;

    vec2  px = mbg_square_to_px(hemi_oct_encode(normalize(cl)), res);
    ivec2 t  = clamp(ivec2(px), ivec2(0), ivec2(int(res) - 1));
    uint  key = s_vis[uint(t.y) * res + uint(t.x)];
    if (key == MBG_EMPTY) return 1.0;                            // nothing in the way
    if (tris[mbg_key_tri(key)].emission.w != 0.0) return 1.0;     // an emitter won

    // Both distances along this texel's centre direction, so the comparison is
    // between two numbers the same rasterizer produced for the same ray.
    vec3  w   = mbg_to_world(mbg_px_to_dir(vec2(t) + vec2(0.5), res));
    float den = dot(tr.n.xyz, w);
    if (abs(den) < 1e-12) return 0.0;
    float te = dot(tr.n.xyz, tr.p0.xyz - g_P) / den;
    if (te <= 0.0) return 0.0;

    float step = 1.0 / (65535.0 * inv_far);
    float stored = float(key >> 16u) * step;
    return stored + 2.0 * step >= te * length(w) ? 1.0 : 0.0;
}

#endif
