#ifndef MBG_COUNTERS_GLSL
#define MBG_COUNTERS_GLSL

// ---------------------------------------------------------------------------
// What the traversal actually touched.
//
// The sweep line has always printed "triangle-rasters", and that number is
// `cameras x scene.count()` -- every camera against every triangle, computed on
// the host before anything runs. On Cornell, where the cull is disabled below
// eight clusters, it happens to be exact. On anything with a cluster hierarchy
// it is an UNCULLED UPPER BOUND and says nothing about how well the cull, the
// distance bound or the traversal order did. Every one of those is a lever this
// example pulls, and none of them moved that number.
//
// Which matters more here than it would elsewhere, because finding 22 settled
// that llvmpipe cannot rank these loops: it put two versions within 3x that an
// RTX 3060 measured 124x apart. The clock in this container is not evidence. A
// count is: it is exact, it is identical on every machine, and it is the thing
// implementation.md's performance section already tells the reader to scale
// with ("READ THE WORK COLUMN, NOT THE TIME COLUMN").
//
// So this is the instrument the work column deserves. Each slot is a 64-bit
// count kept as two 32-bit words, because a sweep tests on the order of 1e10
// texel-triangle pairs and GLSL has no portable 64-bit atomic. The carry is the
// standard unsigned-overflow test: a sum that came out smaller than its addend
// wrapped exactly once, since the addend is itself a 32-bit value.
//
// COST WHEN OFF. `u_count` is a uniform, so every branch here is workgroup
// uniform -- one predicted-not-taken test per loop iteration, no divergence, no
// memory traffic. The counters are accumulated in REGISTERS through the whole
// traversal and pushed to global memory once per workgroup, so the atomics cost
// one contended add per counter per camera rather than one per triangle.
// ---------------------------------------------------------------------------

uniform uint u_count;         // 1 = tally traversal work into MbgCount

#define MBG_CT_CAMERA    0u   // workgroups that ran a traversal
#define MBG_CT_GRP_TEST  1u   // coarse groups distance-tested
#define MBG_CT_GRP_ENTER 2u   // ... and entered
#define MBG_CT_CLU_TEST  3u   // clusters distance-tested
#define MBG_CT_CLU_ENTER 4u   // ... and entered
#define MBG_CT_TRI_SETUP 5u   // triangles FETCHED and set up -- the bandwidth number
#define MBG_CT_TEX_TEST  6u   // texel-triangle pairs tested -- the ALU number
#define MBG_CT_TEXEL     7u   // texels actually rasterized, the live denominator
// A PROBE, not a feature. The distance bound is the only thing pruning an
// entered cluster, and distance is not direction: a cluster can be near and lie
// nowhere near any direction this thread is asking about. An angular test is
// the missing axis -- but any real one needs the texels a thread owns to be
// angularly coherent, and they are strided across the whole target, which is
// finding 24's negative result. So before restructuring anything, count what an
// angular test WOULD reject. Both are measured against a bounding sphere, which
// is conservative: a real test could only reject more.
#define MBG_CT_CLU_ANG   8u   // entered clusters NO live texel's cone can reach
#define MBG_CT_TEX_ANG   9u   // (cluster, live texel) pairs the cone rejects
#define MBG_CT_CLU_PAIR 10u   // ... out of this many such pairs
// A CONSERVATIVENESS ASSERTION, not a statistic. The angular test claims that a
// direction missing a cluster's bounding sphere cannot hit any triangle in it.
// This counts the times a hit was found anyway. It must be zero, and if it is
// not, the test is rejecting geometry and the cull is wrong.
#define MBG_CT_ANG_VIOL 11u
// The LIGHT VIEW is a second traversal, and it has had none of the work the
// hemisphere got. Its occlusion loop walks every cluster lv_cull's frustum let
// through, for every texel, with no per-texel box test of any kind. These two
// ask the finding-30 question about it: how many (cluster, texel) pairs could a
// slab test reject? Probe only -- nothing acts on them.
#define MBG_CT_LV_PAIR  12u
#define MBG_CT_LV_ANG   13u
#define MBG_CT_SLOTS    14u

layout(std430, binding = 14) buffer MbgCount { uint counters[]; };

// The per-invocation tally. Registers, not shared memory: the reduction below
// is one atomic per slot per workgroup either way, and a shared array would cost
// a barrier the traversal does not otherwise need.
uint g_tally[MBG_CT_SLOTS];

void mbg_count_init() {
    for (uint i = 0u; i < MBG_CT_SLOTS; ++i) g_tally[i] = 0u;
}

// Guarded by the caller where it sits in a hot loop, so the test is visible at
// the call site rather than hidden behind a function call.
#define MBG_TALLY(slot, n)  { if (u_count != 0u) g_tally[slot] += (n); }

// One 64-bit add, as two 32-bit words with a carry.
void mbg_count_add64(uint slot, uint v) {
    if (v == 0u) return;
    uint old = atomicAdd(counters[slot * 2u], v);
    if (old + v < old) atomicAdd(counters[slot * 2u + 1u], 1u);
}

// Push this invocation's tally to global memory and ZERO it, so a kernel that
// has two instrumented traversals in it -- the hemisphere and then a light view
// per emitter -- can flush after each without counting the first one twice.
void mbg_count_flush() {
    if (u_count == 0u) return;
    for (uint i = 0u; i < MBG_CT_SLOTS; ++i) {
        mbg_count_add64(i, g_tally[i]);
        g_tally[i] = 0u;
    }
}

#endif
