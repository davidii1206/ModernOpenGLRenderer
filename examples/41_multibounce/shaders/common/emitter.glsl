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
// Both take their VISIBILITY from a rasterized depth sort and nothing else --
// the camera path off its hemisphere (mbg_emitter_mass in raster.glsl), the
// pixel path off a frustum fitted to the emitter (lv_mass in lightview.glsl).
// Both call the magnitude function below, so the two cannot drift apart on the
// half that carries the energy.
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
        // atan2(|a x b|, a . b), NOT acos(a . b), and the difference is
        // measurable. acos is ill-conditioned exactly where this formula spends
        // its time: its derivative is -1/sqrt(1-x^2), so as adjacent edges of a
        // polygon seen from near its own plane approach parallel, the angle
        // loses most of its significant bits. atan2 is well conditioned over the
        // whole range, and both of its arguments are already sitting here.
        //
        // Worth 1.67e-4 to 2.7e-6 on the `closed` gate's worst orientation --
        // a camera sealed inside an emitter, where twelve of these terms have to
        // cancel to exactly PI*L. That gate failed on NVIDIA and passed on
        // llvmpipe purely because the two libraries round acos differently near
        // the ends of its range.
        sum += atan(len, dot(a, b)) * dot(nrm, x / len);
    }
    return tr.emission.xyz * emissive_scale * (0.5 * abs(sum));
}


#endif
