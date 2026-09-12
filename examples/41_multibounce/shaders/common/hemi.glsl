#ifndef MBG_HEMI_GLSL
#define MBG_HEMI_GLSL

// ---------------------------------------------------------------------------
// The hemi-octahedral secondary camera target, and why triangles stay straight
// in it.
//
// Design doc section 4.3 picks a hemi-octahedral map into one square target and
// then says: "Octahedral mapping is nonlinear, so triangle edges are technically
// curved. With LOD holding triangles near pixel size the error stays well under
// a pixel and can be ignored."
//
// That is true of the map taken as a whole and FALSE per quadrant, which is what
// a rasterizer actually needs -- and the distinction is the difference between a
// reference that can be trusted and one that cannot. This example rasterizes the
// full-resolution mesh, where a Cornell wall is two triangles each covering a
// quarter of the hemisphere, so "edges are nearly straight" was never available.
//
// Write the map out. hemi_oct_encode (oct.glsl) is
//
//     L = |d.x| + |d.y| + d.z          (d.z >= 0 on the hemisphere)
//     p = d.xy / L
//     e = (p.x + p.y, p.x - p.y)
//
// Inside a region of fixed sign(d.x), sign(d.y) the denominator L is a LINEAR
// functional of d, so d -> e is d -> (A d)/(c . d): a projective map of the
// direction. Projective maps take lines to lines, and a triangle edge seen from
// a point is a great circle -- a line in the projective sense. So within one
// quadrant the image of a triangle edge is EXACTLY a straight segment, at any
// triangle size, with no small-triangle assumption anywhere.
//
// The four quadrants are separated by d.x = 0 and d.y = 0, themselves great
// circles, which land on the two diagonals of the square. So the recipe is:
// clip the triangle in 3D against z >= 0 and against the two fold planes, then
// project each piece and rasterize it with ordinary straight-edge functions.
// Each quadrant is a projective view exactly the way a hemicube face is; the
// square just packs four of them into one target instead of five faces into
// five (section 4.3's reason for preferring it).
//
// Skipping the fold clip is what actually produces curved edges: a triangle
// spanning a fold would be projected as one straight-edged polygon through a
// map that is only piecewise projective, and the error is unbounded -- a wall
// triangle straddling the +Z axis comes out visibly bent, which is a visibility
// error, not a shading one.
// ---------------------------------------------------------------------------

#include "oct.glsl"

// Texel grid coordinates in [0, res] -> UNNORMALIZED direction in the camera's
// tangent frame. Unnormalized is deliberate: every consumer either normalizes
// once itself or works with ratios where the length cancels.
vec3 mbg_px_to_dir(vec2 px, uint res) {
    vec2 e = px / float(res) * 2.0 - 1.0;
    vec2 t = vec2(e.x + e.y, e.x - e.y) * 0.5;
    return vec3(t, 1.0 - abs(t.x) - abs(t.y));
}

// Tangent-frame vector -> square coordinates, for the quadrant with the given
// component signs. `v` need not be normalized and need not be unit length; only
// its direction matters. The caller must have clipped `v` into that quadrant, so
// that sx*v.x and sy*v.y are both >= 0, and must reject L <= 0.
vec2 mbg_project(vec3 v, float sx, float sy, out float L) {
    L = sx * v.x + sy * v.y + v.z;
    vec2 p = v.xy / L;
    return vec2(p.x + p.y, p.x - p.y);
}

// Square coordinates in [-1,1]^2 -> texel grid coordinates in [0,res].
vec2 mbg_square_to_px(vec2 e, uint res) { return (e * 0.5 + 0.5) * float(res); }

// --- Polygon clipping against a plane THROUGH THE ORIGIN --------------------
//
// Every clip this rasterizer needs is a plane through the camera origin (the
// horizon z = 0 and the two folds x = 0, y = 0), which is the one case where
// Sutherland-Hodgman needs no homogeneous coordinates: the origin is the centre
// of projection, so a plane through it is a great circle in direction space.
//
// MBG_CLIP_MAX is 8: a triangle is 3, the horizon clip adds at most one vertex,
// and each of the two fold clips adds at most one more.
#define MBG_CLIP_MAX 8

int mbg_clip_plane(vec3 src[MBG_CLIP_MAX], int n, vec3 pn, out vec3 dst[MBG_CLIP_MAX]) {
    int m = 0;
    for (int i = 0; i < n; ++i) {
        vec3 a = src[i];
        vec3 b = src[i + 1 == n ? 0 : i + 1];
        float da = dot(pn, a);
        float db = dot(pn, b);
        if (da >= 0.0 && m < MBG_CLIP_MAX) dst[m++] = a;
        if ((da >= 0.0) != (db >= 0.0) && m < MBG_CLIP_MAX) {
            float t = da / (da - db);
            dst[m++] = a + (b - a) * t;
        }
    }
    return m;
}

// --- The packed depth key ---------------------------------------------------
//
// Section 5.3 wants a shared uint64 with depth in the high bits and cluster +
// triangle ID in the low bits, resolved with atomicMin. GLSL has no portable
// 64-bit shared atomic (NV_shader_atomic_int64 and its AMD counterpart are both
// vendor extensions), so this example packs into 32:
//
//     key = (depth16 << 16) | triangle_index16
//
// which caps the scene at 65535 triangles (asserted on the host) and quantizes
// depth to 1/65535 of the far distance. Both are fine here and neither is
// fundamental: the same kernel with a uint64 key and a wider ID field is the
// production form. Ties break to the lowest triangle index, so the winner is
// fully determined by the atomic and the image is bit-reproducible across runs.
#define MBG_EMPTY 0xFFFFFFFFu

uint mbg_pack_key(float dist, float inv_far, uint tri) {
    uint d = uint(clamp(dist * inv_far, 0.0, 1.0) * 65535.0 + 0.5);
    return (d << 16) | (tri & 0xFFFFu);
}

uint mbg_key_tri(uint key) { return key & 0xFFFFu; }

#endif
