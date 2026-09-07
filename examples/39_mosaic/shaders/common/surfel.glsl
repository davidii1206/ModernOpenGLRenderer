#ifndef MOSAIC_SURFEL_GLSL
#define MOSAIC_SURFEL_GLSL

#include "oct.glsl"

// Packed 16-byte object-space surfel record. Mirrors PackedSurfel in
// surfel_bake.hpp; keep the two in sync.
//
//   .x = pos_x_q16 << 16 | radius_q16
//   .y = pos_y_q16 << 16 | pos_z_q16
//   .z = albedo R8 G8 B8 | flags A8
//   .w = octahedral normal, snorm16 pair
//
// Positions are quantized to the MESH aabb and the radius to the set's own
// radius_scale, so both decoders need the set's parameters, not global ones.

const uint SURFEL_EMISSIVE  = 1u;
const uint SURFEL_TWO_SIDED = 2u;

vec3 surfel_position(uvec4 s, vec3 aabb_min, vec3 aabb_extent) {
    vec3 q = vec3(float(s.x >> 16u), float(s.y >> 16u), float(s.y & 0xFFFFu));
    return aabb_min + aabb_extent * (q / 65535.0);
}

float surfel_radius(uvec4 s, float radius_scale) {
    return float(s.x & 0xFFFFu) / 65535.0 * radius_scale;
}

vec3 surfel_albedo(uvec4 s) {
    return vec3(float(s.z >> 24u),
                float((s.z >> 16u) & 0xFFu),
                float((s.z >> 8u) & 0xFFu)) / 255.0;
}

uint surfel_flags(uvec4 s) { return s.z & 0xFFu; }

vec3 surfel_normal(uvec4 s) { return oct_unpack_snorm16(s.w); }

#endif
