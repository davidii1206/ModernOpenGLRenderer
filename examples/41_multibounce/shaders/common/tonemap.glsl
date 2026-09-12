#ifndef SGI_TONEMAP_GLSL
#define SGI_TONEMAP_GLSL

// ---------------------------------------------------------------------------
// View transforms.
//
// The comparison against the path-traced references has to happen in DISPLAY
// space, because both references are already tonemapped -- a linear-space
// comparison would need us to invert a curve we do not have.
//
// Both PNGs are Blender Cycles renders (their tEXt chunks say
// cycles.ViewLayer.samples 4096). The emitter's radiance is
// 17 * (1, 0.7059, 0.2353) = (17.0, 12.0, 4.0), a strongly saturated orange, and
// the reference renders the panel at (254, 250, 238): near-neutral and NOT
// clipped. Desaturating a bright saturated colour toward white without clipping
// it is the signature of Blender 4.x's default AgX view transform.
//
// Example 40 left this as a known gap and called an AgX port the follow-up.
// This is that follow-up, and it was not cosmetic: with ACES the diff against
// the direct reference is a broad systematic brightening of every lit surface,
// which swamps the transport error the comparison is supposed to measure. Under
// AgX the same render's RMSE drops by a third and what is left is structure
// rather than a tone curve.
//
// The implementation is the standard AgX approximation (Troy Sobotka's transform
// as fitted by Benjamin Wrensch and shipped in three.js): inset matrix, log2
// encode over [-12.47, +4.03] EV, a 6th-order contrast sigmoid, outset matrix,
// and a 2.2 decode back to linear so the sRGB OETF below can do the encode. It
// is an approximation of Blender's OCIO transform, not a port of it, and it
// assumes the default look (Blender 4.x "AgX" with look None).
// ---------------------------------------------------------------------------

const int SGI_TM_ACES     = 0;   // Narkowicz 2015 fit
const int SGI_TM_REINHARD = 1;
const int SGI_TM_CLAMP    = 2;   // no tone curve at all
const int SGI_TM_FILMIC   = 3;   // Hejl & Burgess-Dawson (carries its own gamma)
const int SGI_TM_AGX      = 4;   // what both reference PNGs were rendered with

vec3 sgi_srgb_oetf(vec3 c) {
    c = clamp(c, vec3(0.0), vec3(1.0));
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055,
               greaterThan(c, vec3(0.0031308)));
}

vec3 agx_contrast(vec3 x) {
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x +
           0.4298 * x2 + 0.1191 * x - 0.00232;
}

vec3 sgi_agx(vec3 x) {
    const mat3 inset = mat3(
        0.856627153315983,  0.0951212405381588, 0.0482516061458583,
        0.137318972929847,  0.761241990602591,  0.101439036467562,
        0.11189821299995,   0.0767994186031903, 0.811302368396859);
    const mat3 outset = mat3(
         1.1271005818144368,  -0.11060664309660323, -0.016493938717834573,
        -0.1413297634984383,   1.157823702216272,   -0.016493938717834257,
        -0.14132976349843826, -0.11060664309660294,  1.2519364065950405);
    const float min_ev = -12.47393, max_ev = 4.026069;

    vec3 c = inset * max(x, vec3(0.0));
    c = log2(max(c, vec3(1e-10)));
    c = clamp((c - min_ev) / (max_ev - min_ev), 0.0, 1.0);
    c = agx_contrast(c);
    c = outset * c;
    // Back to linear: the OETF at the end of sgi_tonemap does the display encode.
    return clamp(pow(max(c, vec3(0.0)), vec3(2.2)), 0.0, 1.0);
}

vec3 sgi_tonemap(vec3 x, int mode) {
    if (mode == SGI_TM_FILMIC) {
        // Already includes an approximate sRGB encode, so it must not be
        // followed by the OETF.
        vec3 t = max(vec3(0.0), x - 0.004);
        return (t * (6.2 * t + 0.5)) / (t * (6.2 * t + 1.7) + 0.06);
    }
    vec3 y;
    if (mode == SGI_TM_AGX) {
        y = sgi_agx(x);
    } else if (mode == SGI_TM_REINHARD) {
        y = x / (1.0 + x);
    } else if (mode == SGI_TM_CLAMP) {
        y = clamp(x, vec3(0.0), vec3(1.0));
    } else {
        const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
        y = clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
    }
    return sgi_srgb_oetf(y);
}

#endif
