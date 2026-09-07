#ifndef SGI_SURFEL_GLSL
#define SGI_SURFEL_GLSL

// ---------------------------------------------------------------------------
// The SoA surfel set of spec section 1.1. Bindings are global to this example
// (SurfelSet::bind in surfels.cpp binds all six), not per-pass.
//
//   0  vec4  pos.xyz, radius       hot: touched for every candidate
//   1  uint  octahedral normal     hot
//   2  uint  albedo RGB8 + flags   cold: bucket winners only
//   3  uint  emission RGB9E5       cold, emitted RADIANCE L_e
//   4  vec4  irradiance E, written this sweep
//   5  vec4  irradiance E, previous sweep
//
// UNITS. emission is RADIANCE; the irradiance buffers hold IRRADIANCE. A
// surfel's outgoing radiance is
//
//     L_out = L_e + albedo * E / PI
//
// and sgi_outgoing() in brdf.glsl is the ONLY place this example forms it, so
// the /PI cannot go missing in one path and not another -- which is the failure
// section 1.1 singles out, because it is consistent and so survives every
// energy check.
// ---------------------------------------------------------------------------

#include "oct.glsl"
#include "pack.glsl"
#include "brdf.glsl"

const float SGI_PI      = 3.14159265358979;
const float SGI_TWO_PI  = 6.28318530717959;
const float SGI_INV_PI  = 0.31830988618379;

const uint SURFEL_EMISSIVE    = 1u;
const uint SURFEL_TWO_SIDED   = 2u;

layout(std430, binding = 0) readonly buffer SurfelPosRad   { vec4 s_pos_rad[]; };
layout(std430, binding = 1) readonly buffer SurfelNormal   { uint s_normal[]; };
layout(std430, binding = 2) readonly buffer SurfelAlbedo   { uint s_albedo[]; };
layout(std430, binding = 3) readonly buffer SurfelEmission { uint s_emission[]; };

vec3  surfel_pos(uint i)      { return s_pos_rad[i].xyz; }
float surfel_radius(uint i)   { return s_pos_rad[i].w; }
vec3  surfel_normal(uint i)   { return oct_unpack_snorm16(s_normal[i]); }
vec3  surfel_albedo(uint i)   { return unpack_rgb8(s_albedo[i]); }
uint  surfel_flags(uint i)    { return s_albedo[i] >> 24u; }

// The material's glTF doubleSided flag. Recorded by the bake, and deliberately
// NOT used to decide whether a surfel emits from its back face.
//
// doubleSided controls raster backface CULLING. It does not say that a diffuse
// surface re-emits its front side's outgoing radiance out of its back, which is
// what `cosE = abs(cosE)` amounts to -- that is a transmitting surface, not a
// two-sided one. CornellBoxOriginal.glb marks all 8 materials doubleSided, so
// keying emission off this flag lights the sealed volume under each box through
// its own lid and leaks light through every wall.
//
// Emission is therefore front-face only. Two-sidedness stays where it belongs:
// OCCLUSION, which uses abs(dot(nj, -w)) unconditionally, so a back-facing
// surfel emits nothing and still blocks -- section 4.1's "single most common
// source of light leaks".
bool surfel_double_sided(uint i) { return (surfel_flags(i) & 2u) != 0u; }
bool surfel_is_emissive(uint i)  { return (surfel_flags(i) & 1u) != 0u; }
vec3  surfel_emission(uint i) { return unpack_rgb9e5(s_emission[i]); }

// Disc area. With the area-preserving radius of surfels.cpp, sum over the set
// equals the scene's triangle area exactly.
float surfel_area(uint i) {
    float r = s_pos_rad[i].w;
    return SGI_PI * r * r;
}

// `E` is the irradiance the caller read from whichever buffer it is iterating
// against. The expression itself lives in brdf.glsl -- see the note there.
vec3 surfel_outgoing(uint i, vec3 E) {
    return sgi_outgoing(surfel_emission(i), surfel_albedo(i), E);
}

#endif
