#ifndef SGI_PACK_GLSL
#define SGI_PACK_GLSL

// --- RGB9E5 -----------------------------------------------------------------
//
// GL_RGB9_E5: three 9-bit mantissas and a shared 5-bit exponent biased by 15.
// Mirrors pack_rgb9e5 / unpack_rgb9e5 in surfels.cpp.
//
// Both halves use the SAME divisor convention: mantissa * 2^(biased - 15 - 9).
// Example 39's GLSL encoder does not -- it forms the divisor from the unbiased
// exponent while its decoder uses the biased one, a factor of 2^15 apart. That
// encoder is dead code in 39 (only unpack_rgb9e5 is ever called there) so it
// never showed, but do not transcribe it.

vec3 unpack_rgb9e5(uint v) {
    float e = exp2(float(int(v >> 27u) - 15 - 9));
    return vec3(float(v & 0x1FFu), float((v >> 9u) & 0x1FFu), float((v >> 18u) & 0x1FFu)) * e;
}

uint pack_rgb9e5(vec3 c) {
    c = clamp(c, vec3(0.0), vec3(65408.0));
    float maxc = max(c.r, max(c.g, c.b));
    if (!(maxc > 1e-9)) return 0u;
    int e = max(-16, int(floor(log2(maxc)))) + 1;
    float d = exp2(float(e - 9));                     // == 2^((e+15) - 15 - 9)
    if (int(round(maxc / d)) == 512) { e += 1; d = exp2(float(e - 9)); }
    uvec3 m = uvec3(clamp(round(c / d), vec3(0.0), vec3(511.0)));
    return m.r | (m.g << 9u) | (m.b << 18u) | (uint(clamp(e + 15, 0, 31)) << 27u);
}

vec3 unpack_rgb8(uint v) {
    return vec3(float(v & 0xFFu), float((v >> 8u) & 0xFFu), float((v >> 16u) & 0xFFu)) * (1.0 / 255.0);
}

#endif
