#ifndef MBG_RASTER_GLSL
#define MBG_RASTER_GLSL

// ---------------------------------------------------------------------------
// The hemisphere rasterizer: one workgroup, one receiver, one LDS visibility
// buffer resolved by atomicMin.
//
// Shared by every pass that needs to know what a point can see -- the secondary
// cameras of the recursion (raster.comp) and the per-pixel direct term
// (direct_pixel.comp). There is exactly one visibility mechanism in this
// renderer and this is it: rasterize, let the depth sort decide, read the
// winner. No rays, no shadow maps, no second structure.
//
// The includer declares `geom[]` and `quad[]`; the two uniforms the rasterizer
// itself needs are declared here, because a GLSL header cannot use a uniform the
// includer declares after it. Set the receiver frame with mbg_set_receiver()
// before rasterizing.
// ---------------------------------------------------------------------------

#include "scene.glsl"
#include "hemi.glsl"
#include "counters.glsl"

uniform uint  u_res;          // hemisphere target edge, texels

// An ablation knob for the traversal ORDER, which is about how fast the
// distance bound tightens rather than about what gets tested.
uniform uint  u_order;        // 1 = visit groups nearest-first
// 1 = the caller only needs to know WHETHER a direction is blocked, not by what.
// See mbg_resolve_vis_coop: it turns the closest-hit search into an any-hit one,
// which the existing distance bound then terminates almost immediately.
uniform uint  u_anyhit;
// THE RADIANCE INTERVAL. A camera resolves the nearest hit in [u_r0, u_r1) and
// reports nothing outside it, which is what lets one hemisphere be decomposed
// into the radial shells a cascade is made of (world-space-radiance-cascades.md
// section 2.6). Half-open by construction: the accept below is `dist < bd[k]`
// and bd[k] starts at u_r1, so a hit exactly at the far edge belongs to the next
// shell up and is counted once.
//
// Initialising the bound to u_r1 rather than 1e18 also makes `worst` finite from
// the first iteration, so both box tests can reject before the traversal has
// found anything. Defaults 0 and 1e18 reproduce the unbounded traversal exactly.
uniform float u_r0;
uniform float u_r1;
#define MBG_ANG_EPS 1e-5
#define MBG_ANG_ABS 1e-6


// The largest target THIS VARIANT may use. Defined by the wrapper that includes
// raster_body.glsl, not here: the production version this comment used to defer
// to -- "compile a variant per tier" -- is what shaders/raster*.comp now are.
shared uint s_vis[MBG_MAX_TEXELS];
shared vec4 s_red[64];
shared vec4 s_red2[64];

// The continuation distribution, for the single-sample path estimator: one
// weight per texel, turned into an inclusive CDF in place. Only raster.comp's
// spawn uses it, and only after the light views are finished. It used to be
// worth saying that it could alias the light view's own buffer and that 4 KB was
// cheaper than the bug class that comes of sharing storage across a barrier --
// but that buffer no longer exists (finding 37).
shared float s_cdf[MBG_MAX_TEXELS];
shared float s_blk[64];

vec3 g_P, g_T, g_B, g_N;      // the camera, set once per workgroup
uint g_cam;                   // this workgroup's camera index, for the overlap probe

// One bitmask per level-1 camera, one bit per cluster: what this camera actually
// descended into. Written straight to global memory under u_overlap -- a shared
// staging mask would be 1 KB that findings 38 and 42 say is not free, and this is
// a diagnostic, not a path anything ships on.
layout(std430, binding = 16) buffer MbgEnteredB { uint entered[]; };

vec3 mbg_to_local(vec3 w) { return vec3(dot(w, g_T), dot(w, g_B), dot(w, g_N)); }
vec3 mbg_to_world(vec3 l) { return g_T * l.x + g_B * l.y + g_N * l.z; }

// --- Culling a hemisphere ----------------------------------------------------
//
// A LIGHT VIEW'S FRUSTUM DOES NOT TRANSFER. lv_cull tests seven half-spaces and
// throws away almost everything, because the sun's cone is a degree wide. A
// hemisphere is 2*pi steradians: the only hard constraint is the receiver's own
// plane, and standing on Sponza's floor that keeps most of the building.
//
// So the hemisphere gets two weaker levers instead, and needs both:
//
//   A BITMASK, NOT A LIST. lv_cull compacts survivors into 256 slots, which is
//   right when almost nothing survives and useless here, where almost
//   everything does -- the list would overflow on every camera and fall back to
//   walking the scene. One bit per cluster is 4098 bits for Sponza, so the
//   whole answer is 512 bytes of shared memory and cannot overflow at all.
//
//   A DISTANCE BOUND PER TEXEL. The real lever. The texel loop keeps the
//   nearest hit so far in a register, so a cluster whose closest point is
//   already farther than that cannot win and its 64 triangles can be skipped
//   for six operations. This is a BVH's early-out without the tree: the bound
//   tightens as the texel finds nearer geometry, and in an interior the first
//   wall it hits shuts out most of the building.
#define MBG_CLUSTER_WORDS 256            // 8192 clusters = 524k triangles
shared uint s_climask[MBG_CLUSTER_WORDS];
shared uint s_cli_all;                   // 1 = mask unusable, walk everything

void mbg_cull(uint tid, uint stride, uint cluster_count) {
    // Nothing to gain on a scene with a handful of clusters, and the mask clear
    // alone cost Cornell 30% of its sweep before this early-out: 256 words
    // cleared per camera, 1.3 million cameras, to describe 32 triangles.
    if (u_cull == 0u || cluster_count <= 8u ||
        cluster_count > MBG_CLUSTER_WORDS * 32u) {
        if (tid == 0u) s_cli_all = 1u;
        barrier();
        return;
    }
    if (tid == 0u) s_cli_all = 0u;
    // Only the words this scene actually uses.
    uint words = (cluster_count + 31u) >> 5u;
    for (uint w = tid; w < words; w += stride) s_climask[w] = 0u;
    barrier();
    for (uint c = tid; c < cluster_count; c += stride) {
        vec3 lo = clusters[c].lo.xyz - g_P;
        vec3 hi = clusters[c].hi.xyz - g_P;
        vec3 ctr = (lo + hi) * 0.5, e = (hi - lo) * 0.5;
        // Entirely below the receiver's tangent plane: no direction of this
        // hemisphere can reach it.
        if (dot(g_N, ctr) + dot(abs(g_N), e) < 0.0) continue;
        atomicOr(s_climask[c >> 5u], 1u << (c & 31u));
    }
    barrier();
}

bool mbg_cl_live(uint c) {
    return s_cli_all != 0u || (s_climask[c >> 5u] & (1u << (c & 31u))) != 0u;
}

// --- Front to back ------------------------------------------------------------
//
// The distance bound is the only reason the two box levels are worth anything:
// a box farther than the nearest hit so far costs six operations instead of 64
// triangles. But the bound only tightens when the traversal FINDS something
// near, and the order it walks in is the Morton order of the scene -- a property
// of the building, not of the receiver standing in it. A Z-curve crosses from
// one end of Sponza to the other and back several times, so the bound is still
// loose most of the way through and the early-out fires late.
//
// Visiting the groups nearest-first costs one sort per camera over a list that
// is 65 long on Sponza, and the bound is near its final value after the first
// few. This is NOT a hierarchy: it adds no level, stores nothing new, and
// changes no test. It changes the order in which the existing level is visited.
#define MBG_MAX_GROUPS (MBG_CLUSTER_WORDS * 32 / 64)
shared float s_gdist[MBG_MAX_GROUPS];
shared uint  s_gord[MBG_MAX_GROUPS];
shared uint  s_gsorted;                  // 0 = walk them in index order

void mbg_order_groups(uint tid, uint stride, uint group_count, uint super_count) {
    // Nothing to order; or more groups than the array holds, which is only
    // reachable with culling off, where there is no bound to tighten anyway.
    //
    // AND NOT ONCE THERE IS MORE THAN ONE SUPER-GROUP. This permutes the GLOBAL
    // group list, and that stays a permutation of what the traversal walks only
    // while one super-group spans every group. Above that the walk is a sequence
    // of contiguous ranges, and a global permutation would carry groups across
    // their boundaries. Nothing is lost: ordering needs 3..128 groups to engage
    // at all, which at kSuperSize 64 is at most two super-groups, so the one
    // scene it was ever active on -- Cornell+bunny, 34 groups -- still has
    // exactly one and is unaffected.
    if (u_order == 0u || super_count > 1u ||
        group_count <= 2u || group_count > MBG_MAX_GROUPS) {
        if (tid == 0u) s_gsorted = 0u;
        barrier();
        return;
    }
    if (tid == 0u) s_gsorted = 1u;
    for (uint g = tid; g < group_count; g += stride) {
        // Closest point of the box to the receiver, componentwise; zero on any
        // axis the receiver is already between lo and hi. Squared, because the
        // comparisons below only need the order and the traversal squares its
        // bound too.
        vec3 q = max(max(groups[g].lo.xyz - g_P, vec3(0.0)),
                     g_P - groups[g].hi.xyz);
        s_gdist[g] = dot(q, q);
    }
    barrier();
    // A rank sort: every group counts how many come before it and writes itself
    // there. O(n^2) against a list of 65 is about a thousand comparisons spread
    // over 64 threads -- cheaper than the 28 barrier-separated passes a bitonic
    // sort of 128 would need, and it is stable, so the Morton order survives as
    // the tie-break among the boxes the receiver is already inside, all of which
    // score exactly zero.
    for (uint g = tid; g < group_count; g += stride) {
        float dg = s_gdist[g];
        uint rank = 0u;
        for (uint h = 0u; h < group_count; ++h)
            if (s_gdist[h] < dg || (s_gdist[h] == dg && h < g)) ++rank;
        s_gord[rank] = g;
    }
    barrier();
}

uint mbg_group_at(uint gi) { return s_gsorted != 0u ? s_gord[gi] : gi; }

// --- The hemisphere, inverted ------------------------------------------------
//
// One texel per thread, each asking which triangle is nearest along its own
// direction, instead of one triangle per thread atomicMin-ing every texel it
// covers. Same depth sort, opposite loop order -- implementation.md finding 22,
// and lv_mass_texel in lightview.glsl for the version that came first.
//
// AND IT DELETES THE OCTAHEDRAL CLIPPING ENTIRELY. Everything this file argues
// about the folds -- that the map is only piecewise projective, that a triangle
// must be cut at x=0, y=0 and the horizon before its edges are straight, that
// the two pieces need an overlap tolerance or a fold texel falls through the
// crack -- is a consequence of PROJECTING. None of it is a property of the
// visibility question. A direction either lies in a triangle's cone or it does
// not, and mbg_cone_contains answers that without a projection, so there is
// nothing to clip and no fold to fall into. mbg_raster_tri and mbg_fill stay for
// the gates to compare against, but nothing in a frame calls them.
//
// s_vis still lives in shared memory, unlike the light view's buffer: the
// quadrature reads only its own texels, but the tent-weighted spawn reads a
// neighbourhood, so the winners have to be visible across the workgroup. What
// goes away is the atomics -- each texel is written once, by the thread that
// owns it.
// --- Setup once per triangle, test once per direction -------------------------
//
// EVERYTHING THE CONE TEST DOES EXCEPT THREE DOT PRODUCTS IS INDEPENDENT OF THE
// DIRECTION. The three edge planes cross(v_k, v_k+1), the triple product that
// orients them, and the triangle's own plane depend only on the triangle and the
// receiver -- and the cooperative traversal below asks the same triangle about
// MBG_TEXELS_PER_THREAD directions in a row. Computing them inside that loop
// does the work four times.
//
// Split apart, a triangle costs three crosses and a triple product ONCE, and
// each direction costs four dot products and a divide. By operation count that
// is about 155 against 268 per triangle at four texels.
//
// The arithmetic is unchanged and in the same order, so this is bit-identical to
// the fused version by construction -- which is the point: it is a pure code
// motion, and any measured difference would be a bug. (NVIDIA's compiler may
// already hoist some of it out of a constant-trip unrolled loop; the split makes
// it certain and costs nothing if it was already happening.)
struct MbgTriSetup {
    vec3  e0, e1, e2;      // the three edge planes through the receiver
    vec3  pn;              // the triangle's plane normal
    float pc;              // and its offset: t = pc / dot(pn, d)
    float sgn;             // orientation; 0 = edge on, subtends nothing
};

MbgTriSetup mbg_tri_setup(MbgTriGeom tr) {
    vec3 v0 = tr.p0.xyz - g_P;
    vec3 v1 = tr.p1.xyz - g_P;
    vec3 v2 = tr.p2.xyz - g_P;

    MbgTriSetup h;
    h.e0 = cross(v0, v1);
    h.e1 = cross(v1, v2);
    h.e2 = cross(v2, v0);
    float o = dot(h.e0, v2);
    h.sgn = abs(o) < 1e-20 ? 0.0 : (o < 0.0 ? -1.0 : 1.0);
    h.pn = tr.n.xyz;
    h.pc = dot(h.pn, v0);
    return h;
}

bool mbg_hit_dir(MbgTriSetup h, vec3 d, out float dist) {
    if (h.sgn == 0.0) return false;
    float b0 = dot(d, h.e0) * h.sgn;
    float b1 = dot(d, h.e1) * h.sgn;
    float b2 = dot(d, h.e2) * h.sgn;
    float tol = -1e-6 * (abs(b0) + abs(b1) + abs(b2) + 1e-30);
    if (b0 < tol || b1 < tol || b2 < tol) return false;

    float den = dot(h.pn, d);
    if (abs(den) < 1e-20) return false;
    float t = h.pc / den;
    if (t <= 0.0) return false;                  // behind the camera
    dist = t;
    return true;
}

bool mbg_tri_hit_tr(MbgTriGeom tr, vec3 d, out float dist) {
    vec3 v0 = tr.p0.xyz - g_P;
    vec3 v1 = tr.p1.xyz - g_P;
    vec3 v2 = tr.p2.xyz - g_P;

    vec3  pn = tr.n.xyz;
    float pc = dot(pn, v0);

    // NO HALF-SPACE RULE HERE, unlike the light view's lv_texel_key.
    //
    // That rule (finding 19) is a deliberate departure from what a ray cast
    // reports: it declares the far side of a plane the receiver sits on to be
    // solid, because a receiver reconstructed from a depth buffer can land
    // microns outside a wall and a ray would then correctly find nothing in the
    // way. The light view needs it, because the sun's shadow is one direction
    // wide and a single leaked pixel is visible.
    //
    // The hemisphere must not have it, because the oracle gate holds this
    // buffer against a CPU ray cast texel for texel, and a rule that ray casting
    // does not share makes them disagree by construction -- 480 texels and half
    // the energy in dispute when it was in here. Nor does it need it: the
    // camera's own triangle is excluded already, because its plane sits `bias`
    // below the origin, so every upper-hemisphere direction gives t < 0.

    if (!mbg_cone_contains(v0, v1, v2, d)) return false;

    float den = dot(pn, d);
    if (abs(den) < 1e-20) return false;
    float t = pc / den;
    if (t <= 0.0) return false;                  // behind the camera
    // The ray parameter, not a distance: every triangle here is compared along
    // the SAME d, so the scale factor |d| is common and cancels.
    dist = t;
    return true;
}

bool mbg_tri_hit(uint ti, vec3 d, out float dist) {
    return mbg_tri_hit_tr(geom[ti], d, dist);
}

// --- Sharing one traversal across a camera's texels --------------------------
//
// The inverted resolve gives every texel its own traversal, and a camera has
// 256 of them sharing ONE ORIGIN. They rediscover the same scene 256 times, and
// they diverge the moment their distance bounds differ, so the triangle reads
// stop being broadcasts and every lane fetches separately. Measured on Sponza:
// 66 ms for a single camera, and neither a coarse cluster level nor a spatial
// sort moved it, because neither addresses the count.
//
// So the workgroup walks the scene ONCE and each thread carries several texels
// through it. A triangle is fetched a single time and tested against all of
// them; the loop over geometry is workgroup uniform, so there is no divergence
// left to lose the broadcast to. The per-texel distance bound survives as a
// PREDICATE inside the innermost loop rather than a branch that reorders the
// traversal -- a thread whose texels are all nearer simply does nothing for
// that triangle.
//
// This is the projecting rasterizer's loop order with none of its costs: one
// triangle against many texels, but resolved by the cone test, so there is no
// projection, no clipping, and no local-memory polygon arrays. The inversion
// was right for a light view, where 64 texels look at one small thing; it is
// the wrong shape for a hemisphere, where 256 texels look at everything.
#define MBG_TEXELS_PER_THREAD 4

// Which of this thread's live directions pass through a box, as a bitmask.
// Shared by both levels of the hierarchy -- the group boxes and the cluster
// boxes inside them -- because the question is identical and only the box
// changes. Origin is g_P; `inv` is the per-texel reciprocal direction, computed
// once per texel rather than once per texel per box.
//
// Every comparison is written so a NaN falls through to ACCEPT: `tn > tf` is
// false on NaN, and accepting costs some triangle tests while rejecting wrongly
// loses geometry.
uint mbg_box_mask(vec3 lo, vec3 hi, uint nk, vec3 inv[MBG_TEXELS_PER_THREAD]) {
    vec3 e = (hi - lo) * MBG_ANG_EPS + vec3(MBG_ANG_ABS);
    vec3 blo = lo - e - g_P;
    vec3 bhi = hi + e - g_P;
    uint m = 0u;
    for (uint k = 0u; k < nk; ++k) {
        vec3 t1 = blo * inv[k];
        vec3 t2 = bhi * inv[k];
        vec3 tlo = min(t1, t2), thi = max(t1, t2);
        float tn = max(max(tlo.x, tlo.y), max(tlo.z, 0.0));
        float tf = min(min(thi.x, thi.y), thi.z);
        if (tn > tf) continue;
        m |= 1u << k;
    }
    return m;
}

// ANY-HIT AT THE TERMINAL LEVEL, AND WHY IT IS EXACT THERE.
//
// A camera that spawns no children uses this buffer for one thing: the
// quadrature, which asks `key == MBG_EMPTY` and adds the sky, and otherwise adds
// `tr.emission * q.w`. With MBG_NEE on, every emissive triangle carries
// emission.w and the quadrature CONTINUES past it -- it is already counted
// analytically -- and every other triangle has emission zero. So whichever
// triangle is found, the contribution is exactly zero, and the only bit that
// survives is whether one was found at all.
//
// Which makes the terminal hemisphere an occlusion query wearing a visibility
// buffer's clothes. Finding the NEAREST blocker there is work whose result is
// discarded. Stopping at the first turns the search from "scan until the bound
// proves nothing nearer exists" into "stop", and the bound machinery already in
// this loop does the rest for free: a finished texel sets its bound to zero, so
// `worst` collapses and every remaining box fails its test in six operations.
//
// It is NOT exact at a level that spawns -- there the nearest hit is the hit
// point a child camera is placed on -- and it is not exact with MBG_NEE off,
// where the quadrature is how emitters are found in the first place. The host
// sets u_anyhit only when both conditions hold.
void mbg_resolve_vis_coop(uint tid, uint stride, uint group_count, uint super_count) {
    uint n = u_res * u_res;
    uint span = stride * MBG_TEXELS_PER_THREAD;

    mbg_count_init();
    if (u_count != 0u && tid == 0u) {
        g_tally[MBG_CT_CAMERA] = 1u;
        g_tally[MBG_CT_TEXEL]  = n;
    }

    mbg_order_groups(tid, stride, group_count, super_count);

    // A THREAD'S FOUR TEXELS ARE STRIDED, AND GIVING IT A 2x2 TILE INSTEAD DOES
    // NOT HELP -- which is worth recording, because the argument for it is good.
    // `worst` is the loosest of the four bounds and gates both box tests, so a
    // thread prunes only as well as its worst texel; four strided texels look at
    // four unrelated parts of the scene, and any one of them seeing open sky
    // never finds a hit and holds the bound at infinity. Adjacent texels would
    // agree far better. Measured on Sponza it is 0.95x -- a small LOSS. The
    // reason is in the SIMT layer rather than the arithmetic: the box test's
    // `continue` is per lane, the inner loop still runs if any lane in the warp
    // wants it, so what governs is the union over 32 lanes and tightening one
    // lane's bound does not shrink that union. See implementation.md finding 24.
    for (uint base = 0u; base < n; base += span) {
        vec3  d[MBG_TEXELS_PER_THREAD];
        vec3  inv[MBG_TEXELS_PER_THREAD];
        float len[MBG_TEXELS_PER_THREAD];
        uint  best[MBG_TEXELS_PER_THREAD];
        float bd[MBG_TEXELS_PER_THREAD];
        float worst = 0.0;                 // this thread's loosest bound

        // A SLOT THIS THREAD DOES NOT OWN MUST NOT HOLD A BOUND.
        //
        // `i` climbs with k, so a thread's slots are valid up to some nk and
        // dead after it -- and at an 8x8 target they are mostly dead: n is 64
        // and so is the stride, so i = k*64 + tid is in range for k = 0 alone
        // and three slots in four are padding. That is the common case, not the
        // edge one. The default schedule puts 40 of its 41 cameras at res 8.
        //
        // Padding used to be initialized to 1e18 like everything else, and
        // `worst` is the MAX over the four. So worst stayed at infinity for the
        // whole traversal, both box tests compared against infinity, and the
        // distance bound -- the lever findings 22 and 24 are built on -- was
        // inert on every camera that had padding. The cull still removed
        // clusters below the tangent plane; nothing else did anything.
        //
        // Dead slots get a bound of ZERO instead, which is the identity for a
        // max: they cannot loosen `worst`, and no box can be nearer than
        // nothing. nk then keeps them out of the inner loop as well.
        uint nk = 0u;
        for (uint k = 0u; k < MBG_TEXELS_PER_THREAD; ++k) {
            uint i = base + k * stride + tid;
            bool live = i < n;
            d[k] = live ? mbg_to_world(mbg_px_to_dir(vec2(float(i % u_res),
                                                          float(i / u_res)) + vec2(0.5), u_res))
                        : vec3(0.0, 0.0, 1.0);
            len[k] = length(d[k]);
            best[k] = MBG_EMPTY;
            bd[k] = live ? u_r1 : 0.0;
            // Reciprocal direction for the slab test below, once per texel
            // rather than once per texel per cluster. A zero component gives an
            // infinity here, which is exactly what the slab test wants: the ray
            // is parallel to that pair of planes and the interval it contributes
            // is the whole line or nothing.
            inv[k] = 1.0 / d[k];
            if (live) nk = k + 1u;
        }
        worst = max(max(bd[0], bd[1]), max(bd[2], bd[3]));

        // THE THIRD LEVEL. Above the groups there was nothing, so every
        // camera-thread scanned the WHOLE group list, every camera -- 2056 boxes
        // on Bistro, which is exactly 4208958/(32*64), and the one term in this
        // kernel that is O(scene). Finding 46's fit put that scan at 68% of
        // Bistro's sweep against 21% of Sponza's.
        //
        // Same two tests as the level below, one level up, for the same reason
        // the group level itself exists.
        for (uint si = 0u; si < super_count; ++si) {
            vec3 sq = max(max(supers[si].lo.xyz - g_P, vec3(0.0)),
                          g_P - supers[si].hi.xyz);
            if (dot(sq, sq) > worst * worst) continue;
            if (u_angular != 0u &&
                mbg_box_mask(supers[si].lo.xyz, supers[si].hi.xyz, nk, inv) == 0u)
                continue;

            uint gfirst = uint(supers[si].lo.w);
            uint glast  = gfirst + uint(supers[si].hi.w);
            for (uint gi = gfirst; gi < glast; ++gi) {
                uint g = mbg_group_at(gi);
                vec3 gq = max(max(groups[g].lo.xyz - g_P, vec3(0.0)),
                              g_P - groups[g].hi.xyz);
                MBG_TALLY(MBG_CT_GRP_TEST, 1u)
                // The loosest bound any of this thread's texels holds: if the whole
                // group is beyond that, none of them can want it.
                if (dot(gq, gq) > worst * worst) continue;
                // AND THE SAME DIRECTION TEST, ONE LEVEL UP. Without this the group
                // level is distance-only, so it admits a group whenever anything in
                // it is near -- and then every one of its 64 clusters is tested
                // individually. Measured before this line existed: 8.8 of 17 groups
                // entered and 481 cluster tests to enter 3.5 of them.
                if (u_angular != 0u &&
                    mbg_box_mask(groups[g].lo.xyz, groups[g].hi.xyz, nk, inv) == 0u)
                    continue;
                MBG_TALLY(MBG_CT_GRP_ENTER, 1u)

                uint cfirst = uint(groups[g].lo.w);
                uint clast  = cfirst + uint(groups[g].hi.w);
                for (uint c = cfirst; c < clast; ++c) {
                    if (!mbg_cl_live(c)) continue;
                    vec3 q = max(max(clusters[c].lo.xyz - g_P, vec3(0.0)),
                                 g_P - clusters[c].hi.xyz);
                    float qd2 = dot(q, q);
                    MBG_TALLY(MBG_CT_CLU_TEST, 1u)
                    if (qd2 > worst * worst) continue;
                    // IS IT IN THE WAY, NOT JUST NEAR ENOUGH.
                    //
                    // The distance bound is the only thing pruning an entered
                    // cluster, and distance is not direction. A texel is ONE
                    // direction -- mbg_px_to_dir at the texel centre, point sampled,
                    // which is exactly what mbg_hit_dir then intersects -- so a
                    // cluster whose bounding sphere that direction misses cannot
                    // contain its winner, however near it is. Measured before it was
                    // built: 98.9% of entered clusters on Cornell+bunny and 90.7% on
                    // Sponza are reachable by NO direction the thread holds.
                    //
                    // Conservative by construction: the sphere contains the AABB and
                    // the AABB contains every triangle of the cluster, so a ray that
                    // misses the sphere misses all 64 of them. The mask then keeps
                    // the survivors out of each other's way -- a cluster one texel
                    // wants no longer costs the other three their hit tests.
                    //
                    // No normalize and no sqrt: |bc x d|^2 > r^2 |d|^2 is the same
                    // rejection as comparing the perpendicular distance against r,
                    // by Lagrange's identity, and d is already unnormalized.
                    uint amask = (1u << nk) - 1u;
                    if (u_angular != 0u || u_count != 0u) {
                        // A SLAB TEST, NOT A BOUNDING SPHERE.
                        //
                        // The sphere was tried first and is wrong: |bc x d| cancels
                        // catastrophically when the ray runs nearly through a distant
                        // cluster's centre -- large products, tiny difference -- which
                        // is precisely where the ray DOES hit. Against a cluster
                        // small enough (the bunny puts 69k triangles in 1086 of them)
                        // the error reaches the radius itself, and the conservativeness
                        // counter caught it: one hit in a cluster the test had
                        // declared unreachable, surviving a 1000x radius inflation
                        // because inflation is not the failure.
                        //
                        // The slab test has no such term. It is also TIGHTER -- the
                        // box, not the sphere around it -- so it rejects strictly
                        // more. `lo`/`hi` are nudged out by MBG_ANG_EPS to cover
                        // mbg_hit_dir's own edge tolerance, which lets a ray count as
                        // hitting a triangle it passes marginally outside.
                        //
                        // Every comparison is written so that a NaN falls through to
                        // ACCEPT: `tn > tf` is false on NaN, and accepting costs 64
                        // triangle tests while rejecting wrongly loses geometry.
                        amask = mbg_box_mask(clusters[c].lo.xyz, clusters[c].hi.xyz,
                                            nk, inv);
                        if (u_angular != 0u && amask == 0u) continue;
                    }
                    uint pmask = amask;
                    if (u_angular == 0u) amask = (1u << nk) - 1u;
                    MBG_TALLY(MBG_CT_CLU_ENTER, 1u)
                    if (u_overlap != 0u)
                        atomicOr(entered[g_cam * u_entered_words + (c >> 5u)],
                                 1u << (c & 31u));

                    uint first = uint(clusters[c].lo.w);
                    uint last  = first + uint(clusters[c].hi.w);
                    MBG_TALLY(MBG_CT_TRI_SETUP, last - first)
                    for (uint t = first; t < last; ++t) {
                        // One fetch, broadcast across the workgroup, reused by every
                        // texel this thread owns.
                        // One fetch and ONE setup, reused by every texel this
                        // thread owns -- see mbg_tri_setup.
                        MbgTriSetup h = mbg_tri_setup(geom[t]);
                        for (uint k = 0u; k < nk; ++k) {
                            if ((amask & (1u << k)) == 0u) continue;
                            // A texel that already has its answer under any-hit is
                            // done; its bound is zero, so the box tests above have
                            // stopped bringing it work, and this catches the boxes
                            // that contain the receiver and so score zero anyway.
                            if (u_anyhit != 0u && best[k] != MBG_EMPTY) continue;
                            if (qd2 > bd[k] * bd[k]) continue;     // predicate, not a branch
                            MBG_TALLY(MBG_CT_TEX_TEST, 1u)
                            float tt;
                            if (!mbg_hit_dir(h, d[k], tt)) continue;
                            // The angular test said this direction cannot reach this
                            // cluster, and here is a hit in it. Must never fire.
                            if (u_count != 0u && (pmask & (1u << k)) == 0u)
                                ++g_tally[MBG_CT_ANG_VIOL];
                            float dist = tt * len[k];
                            if (dist < u_r0) continue;
                            if (u_anyhit != 0u) { bd[k] = 0.0; best[k] = t; continue; }
                            if (dist < bd[k]) { bd[k] = dist; best[k] = t; }
                        }
                    }
                    worst = max(max(bd[0], bd[1]), max(bd[2], bd[3]));
                }
            }
        }

        for (uint k = 0u; k < MBG_TEXELS_PER_THREAD; ++k) {
            uint i = base + k * stride + tid;
            if (i < n) s_vis[i] = best[k];
        }
    }

    mbg_count_flush();
}

// Fill s_vis for this thread's texels. Replaces clear + rasterize + barrier.
void mbg_resolve_vis(uint tid, uint stride, uint group_count, uint super_count, uint tri_count) {
    uint n = u_res * u_res;

    mbg_count_init();
    if (u_count != 0u && tid == 0u) {
        g_tally[MBG_CT_CAMERA] = 1u;
        g_tally[MBG_CT_TEXEL]  = n;
    }

    // A scene small enough not to be culled does not want the cluster
    // indirection either: one global read per texel to describe 32 triangles is
    // 85 million reads across a Cornell sweep, and it cost 30% of it. The branch
    // is workgroup uniform.
    if (s_cli_all != 0u) {
        for (uint i = tid; i < n; i += stride) {
            vec3  d = mbg_to_world(mbg_px_to_dir(vec2(float(i % u_res),
                                                      float(i / u_res)) + vec2(0.5), u_res));
            // The interval is in WORLD units and mbg_tri_hit returns the ray
            // parameter along an unnormalized direction, so the two are only
            // comparable through |d|. This path used to keep the parameter
            // alone, which is why it needs a length here and the culled paths
            // already had one.
            float len_d = length(d);
            uint  best = MBG_EMPTY;
            float bestdist = u_r1;
            for (uint t = 0u; t < tri_count; ++t) {
                MBG_TALLY(MBG_CT_TRI_SETUP, 1u)
                MBG_TALLY(MBG_CT_TEX_TEST, 1u)
                float tt;
                if (!mbg_tri_hit(t, d, tt)) continue;
                float dist = tt * len_d;
                if (dist < u_r0) continue;
                if (u_anyhit != 0u) { best = t; break; }
                if (dist < bestdist) { bestdist = dist; best = t; }
            }
            s_vis[i] = best;
        }
        mbg_count_flush();
        return;
    }

    mbg_order_groups(tid, stride, group_count, super_count);

    for (uint i = tid; i < n; i += stride) {
        vec3  d = mbg_to_world(mbg_px_to_dir(vec2(float(i % u_res), float(i / u_res))
                                             + vec2(0.5), u_res));
        float len_d = length(d);
        uint  best = MBG_EMPTY;
        // A real distance, not the ray parameter, so it can be compared against
        // a cluster's. 1e18 rather than 1e30 because it gets squared below and
        // 1e30 squared is not a float.
        float bestdist = u_r1;

        // Two levels, coarse first. Every texel of this camera shares an origin
        // and so rediscovers the same scene; testing 65 group boxes before 4098
        // cluster boxes turns that from O(n) into O(sqrt n), and the distance
        // bound prunes at both levels.
        for (uint si = 0u; si < super_count; ++si) {
            vec3 sq = max(max(supers[si].lo.xyz - g_P, vec3(0.0)),
                          g_P - supers[si].hi.xyz);
            if (dot(sq, sq) > bestdist * bestdist) continue;

            uint gfirst = uint(supers[si].lo.w);
            uint glast  = gfirst + uint(supers[si].hi.w);
            for (uint gi = gfirst; gi < glast; ++gi) {
                uint g = mbg_group_at(gi);
                vec3 gq = max(max(groups[g].lo.xyz - g_P, vec3(0.0)),
                              g_P - groups[g].hi.xyz);
                MBG_TALLY(MBG_CT_GRP_TEST, 1u)
                if (dot(gq, gq) > bestdist * bestdist) continue;
                MBG_TALLY(MBG_CT_GRP_ENTER, 1u)

                uint cfirst = uint(groups[g].lo.w);
                uint clast  = cfirst + uint(groups[g].hi.w);
                for (uint c = cfirst; c < clast; ++c) {
                    if (!mbg_cl_live(c)) continue;
                    // Closest point of the box to the receiver, componentwise: zero
                    // on any axis the receiver is already between lo and hi.
                    vec3 q = max(max(clusters[c].lo.xyz - g_P, vec3(0.0)),
                                 g_P - clusters[c].hi.xyz);
                    MBG_TALLY(MBG_CT_CLU_TEST, 1u)
                    if (dot(q, q) > bestdist * bestdist) continue;
                    MBG_TALLY(MBG_CT_CLU_ENTER, 1u)

                    uint first = uint(clusters[c].lo.w);
                    uint last  = first + uint(clusters[c].hi.w);
                    MBG_TALLY(MBG_CT_TRI_SETUP, last - first)
                    MBG_TALLY(MBG_CT_TEX_TEST, last - first)
                    for (uint t = first; t < last; ++t) {
                        float tt;
                        if (!mbg_tri_hit(t, d, tt)) continue;
                        float dist = tt * len_d;
                        if (dist < u_r0) continue;
                        // Any-hit: the bound goes to zero, so every remaining box
                        // fails its test and the two loops above unwind themselves.
                        if (u_anyhit != 0u) { best = t; bestdist = 0.0; break; }
                        if (dist < bestdist) { bestdist = dist; best = t; }
                    }
                    if (u_anyhit != 0u && best != MBG_EMPTY) break;
                }
                if (u_anyhit != 0u && best != MBG_EMPTY) break;
            }
            if (u_anyhit != 0u && best != MBG_EMPTY) break;
        }
        s_vis[i] = best;
    }

    mbg_count_flush();
}

// --- The direct term, analytically ------------------------------------------
//
// Doc section 3, non-goals: "No correct area-light soft shadows from the GI
// pass. Direct lighting stays a separate conventional pass." This is that pass,
// kept inside the same kernel because the visibility it needs is already
// sitting in shared memory.
//
// WHY IT IS NOT OPTIONAL HERE. The emitter is the brightest thing in the scene
// by two orders of magnitude and subtends a few percent of the hemisphere, so a
// texel either sees all of it or none of it. At 32x32 that is a few percent of
// error per camera, uncorrelated between neighbours, and it reads as the
// mottling that dominated the first version of this example. At the 8x8 targets
// the deeper levels use, the panel can miss every texel centre of a camera that
// plainly sees it, and the indirect term goes blotchy.
//
// So the emitter's contribution is split in two:
//
//   magnitude   the EXACT projected solid angle of the emitter polygon, clipped
//               to the hemisphere -- Lambert's formula, no quadrature at all
//   visibility  a contribution-weighted fraction, sampled against the depth
//               already in s_vis
//
// The magnitude carries all the energy and is now exact. The visibility is the
// only sampled quantity left, it is bounded in [0,1], and it is 1 or 0 over most
// of the image with a smooth ramp across penumbrae -- which is exactly the error
// budget an area light wants. The hemisphere resolve then skips L_e on these
// triangles (emission.w), so nothing is counted twice.

// Clear, then rasterize a triangle range. `stride` is the workgroup size.
void mbg_clear_vis(uint tid, uint stride) {
    uint texels = u_res * u_res;
    for (uint i = tid; i < texels; i += stride) s_vis[i] = MBG_EMPTY;
}

// 32-bit integer hash (Chris Wellons' triple32-style mix). Only used to pick a
// rotation, so the bar is "decorrelated between neighbours", not randomness.
uint mbg_hash(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// THE TANGENT FRAME IS ROTATED PER RECEIVER, AND THAT IS NOT A DETAIL.
//
// Every receiver sampling the hemisphere with the SAME texel grid means every
// receiver makes the same quadrature error, and an error that all of them make
// together is not noise -- it is a pattern. A scene feature crosses a texel
// boundary at the same place for a whole row of receivers and their sums step
// together, which is why the indirect term came out banded rather than grainy
// (implementation.md, finding 14).
//
// Rotating each receiver's frame by a hash of its own position decorrelates
// them: the same error becomes spatial noise, which anything that averages
// neighbours removes. Example 40 reaches the same conclusion about its own
// microbuffer in one line of common/micro.glsl -- "the coherent octahedral
// banding on a flat wall becomes spatial noise instead of a pattern" -- and is
// the reason it gets clean indirect out of 8x8 buckets.
//
// The hash is over the receiver's POSITION rather than its index, so the
// rotation does not change when the chunk scheduler assigns a camera to a
// different slot; a sweep is reproducible bit for bit either way.
// A uniform in [0,1) for a receiver at `pos`, on an independent stream.
//
// EVERY random number in this renderer comes from here, and every one of them is
// a hash of a POSITION rather than of a camera index or a frame counter. That is
// what keeps a sweep reproducible bit for bit: the chunk scheduler is free to
// assign a camera to a different slot, and a receiver at the same point still
// draws the same numbers. `stream` separates the decisions made at one point --
// the hemisphere's rotation, the sun cone's, which direction a path continues
// through, whether it survives roulette -- so that decorrelating one does not
// lock it to another.
float mbg_rand(vec3 pos, uint stream) {
    uint seed = mbg_hash(stream ^ floatBitsToUint(pos.x) ^
                mbg_hash(floatBitsToUint(pos.y) ^
                mbg_hash(floatBitsToUint(pos.z))));
    return float(seed) * (1.0 / 4294967296.0);
}

float mbg_jitter_angle(vec3 pos, uint salt) {
    return mbg_rand(pos, salt) * 6.28318530717959;
}

// --- The continuation distribution ------------------------------------------
//
// s_cdf holds a non-negative weight per texel on entry and its INCLUSIVE prefix
// sum on exit; the return value is the total, identical in every invocation.
//
// Blocked scan rather than a serial one on thread 0: 64 contiguous runs scanned
// in parallel, a Hillis-Steele scan of the 64 run totals, then one add per
// entry. At 256 texels that is 4 serial adds plus 6 barriers instead of 256
// dependent adds with 63 lanes idle.
float mbg_scan_cdf(uint tid, uint n) {
    uint per = (n + 63u) / 64u;
    uint lo = min(tid * per, n);
    uint hi = min(lo + per, n);

    float run = 0.0;
    for (uint i = lo; i < hi; ++i) { run += s_cdf[i]; s_cdf[i] = run; }
    s_blk[tid] = run;
    barrier();

    for (uint d = 1u; d < 64u; d <<= 1u) {
        float v = tid >= d ? s_blk[tid - d] : 0.0;
        barrier();
        s_blk[tid] += v;
        barrier();
    }
    // s_blk[tid] is the inclusive scan of the run totals, so subtracting this
    // run's own total leaves the offset the run starts at.
    float off = s_blk[tid] - run;
    for (uint i = lo; i < hi; ++i) s_cdf[i] += off;
    float total = s_blk[63];
    barrier();
    return total;
}

// The first index whose inclusive CDF reaches `u`. No barriers: the search is
// per invocation and reads only.
uint mbg_sample_cdf(float u, uint n) {
    uint lo = 0u, hi = n - 1u;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1u;
        if (s_cdf[mid] < u) lo = mid + 1u; else hi = mid;
    }
    return lo;
}

void mbg_set_receiver(vec3 pos, vec3 nrm, float bias, bool jitter) {
    g_N = normalize(nrm);
    mbg_onb(g_N, g_T, g_B);
    g_P = pos + g_N * bias;

    if (jitter) {
        float a = mbg_jitter_angle(pos, 0x27D4EB2Du);
        float c = cos(a), sn = sin(a);
        vec3 t2 = g_T * c + g_B * sn;
        g_B = g_B * c - g_T * sn;
        g_T = t2;
    }
}

#endif
