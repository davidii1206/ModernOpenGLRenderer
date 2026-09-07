#ifndef SGI_OCT_GLSL
#define SGI_OCT_GLSL

// Octahedral direction encoding.
//
// oct_encode/oct_decode map a UNIT SPHERE to [-1,1]^2 and are used for normal
// compression. hemi_oct_* map a UNIT HEMISPHERE about +Z to [-1,1]^2 and are
// used for the microbuffer parametrization -- do not mix the two, they are
// different mappings with different Jacobians.
//
// The sphere mapping matches the convention in gfx/coverage_atlas.cpp and
// examples 32/34/35/37 (the z < 0 fold), so packed normals are interchangeable
// with those examples' data.

vec2 oct_encode(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) n.xy = (1.0 - abs(vec2(n.y, n.x))) * sign(n.xy);
    return n.xy;
}

vec3 oct_decode(vec2 e) {
    vec3 n = vec3(e, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(vec2(n.y, n.x))) * sign(n.xy);
    return normalize(n);
}

// snorm16 pair <-> uint, matching pack_oct_snorm16 in surfels.cpp.
uint oct_pack_snorm16(vec3 n) {
    vec2 e = clamp(oct_encode(n), vec2(-1.0), vec2(1.0));
    ivec2 q = ivec2(round(e * 32767.0));
    return (uint(q.x & 0xFFFF) << 16) | uint(q.y & 0xFFFF);
}

vec3 oct_unpack_snorm16(uint p) {
    int hi = int(p >> 16u);      hi = hi > 32767 ? hi - 65536 : hi;
    int lo = int(p & 0xFFFFu);   lo = lo > 32767 ? lo - 65536 : lo;
    return oct_decode(vec2(float(hi), float(lo)) / 32767.0);
}

// --- Hemispherical octahedral mapping (the microbuffer) --------------------
//
// Maps the +Z hemisphere onto the full [-1,1]^2 square. Unlike the disk
// (Nusselt) projection used by examples 36 and 37, the cosine term does NOT
// cancel here: the resolve must weight each texel by cos(theta) * solid angle
// explicitly. See bf_micro.comp.

vec2 hemi_oct_encode(vec3 d) {
    d /= (abs(d.x) + abs(d.y) + d.z);          // d.z >= 0 on the hemisphere
    return vec2(d.x + d.y, d.x - d.y);         // 45-degree rotation fills the square
}

vec3 hemi_oct_decode(vec2 e) {
    vec2 t = vec2(e.x + e.y, e.x - e.y) * 0.5;
    return normalize(vec3(t, 1.0 - abs(t.x) - abs(t.y)));
}

#endif
