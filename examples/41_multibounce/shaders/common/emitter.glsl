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
//   raster.comp        one receiver per secondary camera. Visibility comes from
//                      the hemisphere buffer it just rasterized. Feeds the bounce
//                      transport, where a soft answer is fine.
//   direct_pixel.comp  one receiver per PIXEL. Visibility comes from shadow rays.
//                      Feeds the image, where it is the sharpest term there is.
//
// Both call the magnitude function below, so the two paths cannot drift apart.
// The triangle buffer and the emitter list are declared by the including shader;
// only the maths lives here.
// ---------------------------------------------------------------------------

#include "hemi.glsl"

vec2 mbg_hammersley(uint i, uint n) {
    uint b = (i << 16u) | (i >> 16u);
    b = ((b & 0x55555555u) << 1u) | ((b & 0xAAAAAAAAu) >> 1u);
    b = ((b & 0x33333333u) << 2u) | ((b & 0xCCCCCCCCu) >> 2u);
    b = ((b & 0x0F0F0F0Fu) << 4u) | ((b & 0xF0F0F0F0u) >> 4u);
    b = ((b & 0x00FF00FFu) << 8u) | ((b & 0xFF00FF00u) >> 8u);
    return vec2((float(i) + 0.5) / float(n), float(b) * 2.3283064365386963e-10);
}

// Moller-Trumbore against one triangle, two-sided -- the rasterizer does not
// cull either. raster.comp calls it on the single triangle its visibility buffer
// already named, as a confirmation; direct_pixel.comp calls it on every triangle
// in the scene, as a search.
bool mbg_ray_tri(vec3 o, vec3 d, MbgTri tr, out float t) {
    vec3 e1 = tr.p1.xyz - tr.p0.xyz;
    vec3 e2 = tr.p2.xyz - tr.p0.xyz;
    vec3 pv = cross(d, e2);
    float det = dot(e1, pv);
    if (abs(det) < 1e-20) return false;
    float inv = 1.0 / det;
    vec3 tv = o - tr.p0.xyz;
    float u = dot(tv, pv) * inv;
    if (u < 0.0 || u > 1.0) return false;
    vec3 qv = cross(tv, e1);
    float v = dot(d, qv) * inv;
    if (v < 0.0 || u + v > 1.0) return false;
    t = dot(e2, qv) * inv;
    return t > 0.0;
}

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


#endif
