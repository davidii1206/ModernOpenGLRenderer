#ifndef MOSAIC_GBUFFER_GLSL
#define MOSAIC_GBUFFER_GLSL

// G-buffer layout shared by the geometry pass and every consumer.
//
//   gbuf0  RGBA8     albedo.rgb            | a: 1 = geometry, 0 = sky
//   gbuf1  RGBA16F   normal.xyz (world)    | w: roughness
//   gbuf2  RGBA16F   emissive.rgb          | w: metallic
//   gbuf3  RG16F     motion vector (NDC, current -> previous)
//   depth  DEPTH32F
//
// World position is RECONSTRUCTED from depth rather than stored: examples 37
// and 38 both kept a full RGBA16F position target, which costs 8 MB/frame of
// bandwidth at 1080p to store something the depth buffer already encodes.

vec3 world_from_depth(float depth01, vec2 uv, mat4 inv_view_proj) {
    vec4 ndc = vec4(uv * 2.0 - 1.0, depth01 * 2.0 - 1.0, 1.0);
    vec4 p = inv_view_proj * ndc;
    return p.xyz / p.w;
}

#endif
