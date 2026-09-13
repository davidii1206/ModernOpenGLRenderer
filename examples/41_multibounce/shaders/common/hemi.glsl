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
// MBG_CLIP_MAX is 10. The hemisphere rasterizer needs 6 (a triangle is 3, the
// horizon clip adds at most one vertex, and each of the two fold clips adds one
// more). The light view needs 8: its near plane adds one and its four side
// planes add one each. 10 leaves two vertices of headroom, which costs two
// vec3s of stack and removes the question of whether a clip that lands exactly
// on a plane can ever emit a duplicate and overrun.
#define MBG_CLIP_MAX 10

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

// --- Coverage without projection --------------------------------------------
//
// The one test both inverted rasterizers share, so the tolerance below lives in
// exactly one place. Does the cone that (v0,v1,v2) subtends from the origin
// contain direction `d`? All three vectors are relative to the viewpoint and
// none of them need be normalized.
//
// The three planes through the viewpoint and the triangle's edges have normals
// cross(v_k, v_k+1), and `d` is inside when all three dot products share the
// sign of the triple product [v0,v1,v2]. The antipodal cone flips all three, so
// it is excluded rather than matched. Under a projective map this expression IS
// the 2D edge function up to a positive scale -- the same inclusive test a
// rasterizer does after projecting, done before.
//
// THE TOLERANCE IS LOAD BEARING. A direction can land exactly on the edge two
// triangles share, and that is the common case rather than a corner one: a quad
// is two triangles split along a diagonal. There all three products are zero
// plus float noise, and if both pieces round the wrong way the direction is
// covered by neither and an opaque surface develops a slit. The `occ` gate
// measured 5.4% of an emitter coming through a blocker that covers it twice
// over, falling as 1/res -- the signature of a defect on a line. Overlap is
// free: both pieces compute the same depth from the same plane.
bool mbg_cone_contains(vec3 v0, vec3 v1, vec3 v2, vec3 d) {
    vec3 e0 = cross(v0, v1);
    vec3 e1 = cross(v1, v2);
    vec3 e2 = cross(v2, v0);
    float o = dot(e0, v2);
    if (abs(o) < 1e-20) return false;          // edge on: subtends nothing
    float sgn = o < 0.0 ? -1.0 : 1.0;
    float b0 = dot(d, e0) * sgn;
    float b1 = dot(d, e1) * sgn;
    float b2 = dot(d, e2) * sgn;
    float tol = -1e-6 * (abs(b0) + abs(b1) + abs(b2) + 1e-30);
    return b0 >= tol && b1 >= tol && b2 >= tol;
}

// --- The visibility key ------------------------------------------------------
//
// A TRIANGLE INDEX, AND NOTHING ELSE. Section 5.3 wants a shared uint64 with
// depth in the high bits and the triangle ID in the low bits, resolved with
// atomicMin, and this example used to pack that into 32 bits -- depth16 << 16 |
// tri16 -- because GLSL has no portable 64-bit shared atomic. That capped the
// scene at 65535 triangles, which the Cornell bunny (69483) already exceeds.
//
// The cap is gone because the ATOMIC is gone. Inverting the rasterizers
// (findings 22 and 23) gave each texel to one thread, so the nearest triangle is
// found by comparing full-precision floats in a register and written once. The
// depth never needs to survive into shared memory: every consumer of this buffer
// -- the quadrature, the spawn, the oracle dump -- asks only which triangle won.
//
// So the key is the index, the sentinel is the all-ones word, and the scene
// limit is 4 billion triangles rather than 65 thousand. No extension needed.
#define MBG_EMPTY 0xFFFFFFFFu

// Every pass that measures a distance needs the same far reference, so it is
// declared here rather than in one of the rasterizers.
uniform float u_inv_far;

#endif
