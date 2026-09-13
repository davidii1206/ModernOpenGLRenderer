#ifndef SGI_FMM_GLSL
#define SGI_FMM_GLSL

// The FMM tree, in ONE storage buffer.
//
// Four arrays would be the natural layout and it does not fit:
// GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS is 16 on this hardware and bf_micro.comp
// was already at the limit, so a second FMM binding cost it the whole shader.
// One buffer, four regions, offsets as uniforms:
//
//   [0, cell_off)                      slot map, dense per level: slot or -1
//   [cell_off, ilm_off)                linear cell index of each slot
//   [ilm_off, lsh_off)                 multipole,       27 words per slot
//   [lsh_off, ...)                     local expansion, 27 words per slot
//
// Words are uint and reinterpreted, because a storage block has one type and the
// slot map is signed while the coefficients are float. uintBitsToFloat is free.

#include "sh2.glsl"

layout(std430, binding = 12) buffer FmmTree { uint fmm_w[]; };

uniform uint u_fmm_cell_off;
uniform uint u_fmm_ilm_off;
uniform uint u_fmm_lsh_off;

ivec3 sgi_fmm_coord(uint c, ivec3 res) {
    const uint x = c % uint(res.x);
    const uint t = c / uint(res.x);
    return ivec3(int(x), int(t % uint(res.y)), int(t / uint(res.y)));
}

uint sgi_fmm_linear(ivec3 c, ivec3 res) {
    return uint(c.x) + uint(res.x) * (uint(c.y) + uint(res.y) * uint(c.z));
}

// -1 when the coordinate is outside the level or its cell is empty.
int sgi_fmm_slot_at(ivec3 c, ivec3 res, uint slot_base) {
    if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, res))) return -1;
    return int(fmm_w[slot_base + sgi_fmm_linear(c, res)]);
}

uint sgi_fmm_cell(uint coeff_base, uint s) {
    return fmm_w[u_fmm_cell_off + coeff_base + s];
}

void sgi_fmm_load(inout vec3 c[9], uint region, uint base, uint s) {
    const uint o = region + (base + s) * 27u;
    for (uint k = 0u; k < 9u; ++k)
        c[k] = vec3(uintBitsToFloat(fmm_w[o + k * 3u + 0u]),
                    uintBitsToFloat(fmm_w[o + k * 3u + 1u]),
                    uintBitsToFloat(fmm_w[o + k * 3u + 2u]));
}

void sgi_fmm_store(vec3 c[9], uint region, uint base, uint s) {
    const uint o = region + (base + s) * 27u;
    for (uint k = 0u; k < 9u; ++k) {
        fmm_w[o + k * 3u + 0u] = floatBitsToUint(c[k].r);
        fmm_w[o + k * 3u + 1u] = floatBitsToUint(c[k].g);
        fmm_w[o + k * 3u + 2u] = floatBitsToUint(c[k].b);
    }
}

void sgi_fmm_add(vec3 c[9], uint region, uint base, uint s) {
    const uint o = region + (base + s) * 27u;
    for (uint k = 0u; k < 9u; ++k) {
        fmm_w[o + k * 3u + 0u] = floatBitsToUint(uintBitsToFloat(fmm_w[o + k*3u+0u]) + c[k].r);
        fmm_w[o + k * 3u + 1u] = floatBitsToUint(uintBitsToFloat(fmm_w[o + k*3u+1u]) + c[k].g);
        fmm_w[o + k * 3u + 2u] = floatBitsToUint(uintBitsToFloat(fmm_w[o + k*3u+2u]) + c[k].b);
    }
}

#endif
