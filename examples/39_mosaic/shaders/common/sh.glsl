#ifndef MOSAIC_SH_GLSL
#define MOSAIC_SH_GLSL

// Order-1 (4 coefficient) spherical harmonics for RGB irradiance.
//
// Nothing in this repository had SH before, so the conventions are stated
// explicitly here rather than assumed:
//
//   Y_0   = 0.282095                 (constant)
//   Y_1-1 = 0.488603 * y
//   Y_10  = 0.488603 * z
//   Y_11  = 0.488603 * x
//
// project_radiance() accumulates INCOMING RADIANCE weighted by the differential
// solid angle. Evaluating irradiance from those coefficients then needs the
// Ramamoorthi & Hanrahan convolution constants A0 = pi, A1 = 2pi/3, which fold
// the cosine lobe into the basis. Skipping that step is the classic way to get a
// plausible-looking but systematically wrong result, so eval_irradiance() bakes
// them in and callers never touch raw coefficients.

const float SH_C0 = 0.282095;
const float SH_C1 = 0.488603;

// A1/A0 ratio, premultiplied so eval_irradiance is a single dot product.
const float SH_A0 = 3.14159265;
const float SH_A1 = 2.0943951;   // 2*pi/3

struct SH1 {
    vec3 c[4];   // DC, y, z, x
};

SH1 sh_zero() {
    SH1 s;
    s.c[0] = vec3(0.0); s.c[1] = vec3(0.0); s.c[2] = vec3(0.0); s.c[3] = vec3(0.0);
    return s;
}

void sh_accumulate(inout SH1 s, vec3 dir, vec3 radiance, float d_omega) {
    vec3 w = radiance * d_omega;
    s.c[0] += w * SH_C0;
    s.c[1] += w * (SH_C1 * dir.y);
    s.c[2] += w * (SH_C1 * dir.z);
    s.c[3] += w * (SH_C1 * dir.x);
}

SH1 sh_scale(SH1 s, float k) {
    SH1 o;
    for (int i = 0; i < 4; ++i) o.c[i] = s.c[i] * k;
    return o;
}

SH1 sh_lerp(SH1 a, SH1 b, float t) {
    SH1 o;
    for (int i = 0; i < 4; ++i) o.c[i] = mix(a.c[i], b.c[i], t);
    return o;
}

// Cosine-convolved irradiance in direction n, divided by pi so the result is
// the value to multiply albedo by directly (i.e. outgoing radiance for a
// Lambertian surface, not irradiance in W/m^2).
//
// DE-RINGED. An L1 basis cannot represent a strongly directional distribution,
// and the reconstruction
//
//     E(n) = A0*C0*c0 + A1*C1*(c1 . n)
//
// dips NEGATIVE wherever the L1 term outweighs the DC term -- which is exactly
// what a single bright emitter such as a Cornell ceiling panel produces. Simply
// clamping the result to zero turns that dip into a hard-edged black patch with
// a curved boundary, and those patches move as the light does.
//
// The minimum over all n is A0*C0*c0 - A1*C1*|c1|, so scaling the whole L1 band
// by the largest factor that keeps that minimum at zero removes the ringing
// while preserving both the DC term (total energy) and the direction of the
// gradient. Windowing per channel rather than on luminance keeps a strongly
// coloured bounce from being de-saturated by its brightest channel.
vec3 sh_eval_irradiance(SH1 s, vec3 n) {
    vec3 dc = SH_A0 * SH_C0 * s.c[0];
    vec3 l1 = SH_A1 * SH_C1 * (s.c[1] * n.y + s.c[2] * n.z + s.c[3] * n.x);

    // Per-channel magnitude of the L1 gradient, i.e. the worst-case dip.
    vec3 mag = SH_A1 * SH_C1 * vec3(
        length(vec3(s.c[3].r, s.c[1].r, s.c[2].r)),
        length(vec3(s.c[3].g, s.c[1].g, s.c[2].g)),
        length(vec3(s.c[3].b, s.c[1].b, s.c[2].b)));

    vec3 window = min(vec3(1.0), dc / max(mag, vec3(1e-8)));
    return max((dc + l1 * window) / SH_A0, vec3(0.0));
}

// --- Storage ---------------------------------------------------------------
//
// Four vec4 per SOURCE surfel (not per live surfel: S0 reassigns live indices
// every frame, so a cache keyed on the live index would scramble whenever LOD
// selection or instance ordering changed). Four RGB coefficients need 12 floats;
// the four spare lanes carry the filter state the temporal pass needs.
//
//   [0] = (c0.rgb, variance)
//   [1] = (c1.rgb, age)
//   [2] = (c2.rgb, last_direct_luma)
//   [3] = (c3.rgb, unused)
//
// Deliberately NOT the spec's packed RGB9E5 + snorm16-relative form: the packed
// version costs precision in the temporal EMA and a pack/unpack round trip in
// every stage that touches it, to save memory that is not the bottleneck here.

const uint SH_STRIDE = 4u;

SH1 sh_load(vec4 v0, vec4 v1, vec4 v2, vec4 v3) {
    SH1 s;
    s.c[0] = v0.xyz;
    s.c[1] = v1.xyz;
    s.c[2] = v2.xyz;
    s.c[3] = v3.xyz;
    return s;
}

#endif
