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
// None of the curves below is AgX. That is a KNOWN GAP: it makes the numeric
// diff against the references approximate, most visibly on the panel itself.
// SGI_TONEMAP lets you pick whichever lands closest, and the calibration line
// printed at startup shows what each one does with (17, 12, 4) so the choice is
// measured rather than guessed. An exact AgX port is the follow-up.
// ---------------------------------------------------------------------------

const int SGI_TM_ACES     = 0;   // Narkowicz 2015 fit
const int SGI_TM_REINHARD = 1;
const int SGI_TM_CLAMP    = 2;   // no tone curve at all
const int SGI_TM_FILMIC   = 3;   // Hejl & Burgess-Dawson (carries its own gamma)

vec3 sgi_srgb_oetf(vec3 c) {
    c = clamp(c, vec3(0.0), vec3(1.0));
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055,
               greaterThan(c, vec3(0.0031308)));
}

vec3 sgi_tonemap(vec3 x, int mode) {
    if (mode == SGI_TM_FILMIC) {
        // Already includes an approximate sRGB encode, so it must not be
        // followed by the OETF.
        vec3 t = max(vec3(0.0), x - 0.004);
        return (t * (6.2 * t + 0.5)) / (t * (6.2 * t + 1.7) + 0.06);
    }
    vec3 y;
    if (mode == SGI_TM_REINHARD) {
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
