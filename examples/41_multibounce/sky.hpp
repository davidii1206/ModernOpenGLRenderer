#pragma once

// ---------------------------------------------------------------------------
// The sun and the sky dome, host side.
//
// One struct rather than a handful of loose config fields, because four passes
// need the identical numbers -- the secondary cameras (shaders/raster.comp), the
// per-pixel direct term (shaders/direct_pixel.comp) and the display pass's
// background (shaders/display.frag) -- and a sun that is half a degree off
// between the shading and the shadow is a bug nobody would find by looking at
// the image. bind() is the single place the uniforms are named.
//
// The physics and the split between what is analytic and what is rasterized live
// in shaders/common/sky.glsl; this is only the plumbing and the presets.
// ---------------------------------------------------------------------------

#include "gpu_util.hpp"

#include <cmath>

namespace mbg {

struct SkyLight {
    // The constant floor, in every direction including below the horizon. This
    // is the original MBG_SKY: a crude ambient that existed before there was a
    // dome, kept because the full-GI reference comparison is calibrated against
    // it (implementation.md, finding 5) and every measurement in that file
    // assumes it. Zero unless asked for.
    glm::vec3 ambient{0.0f};

    // The dome. Radiance, not colour: these are multiplied by solid angle and
    // integrated, so 1.0 is about as bright as a Lambertian white surface under
    // an emitter of radiance PI.
    glm::vec3 zenith{0.0f};
    glm::vec3 horizon{0.0f};
    glm::vec3 ground{0.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};

    // The sun. `irradiance` is what a surface FACING the sun receives, which is
    // radiance times solid angle -- the one number a user wants to set, and the
    // one that keeps the exposure fixed when the angular radius changes.
    glm::vec3 irradiance{0.0f};
    glm::vec3 dir{0.0f, 1.0f, 0.0f};   // toward the sun; normalized in bind()
    float     angle = 1.2f;            // angular RADIUS, degrees

    bool has_sun() const {
        return irradiance.x > 0.0f || irradiance.y > 0.0f || irradiance.z > 0.0f;
    }
    bool has_dome() const {
        auto any = [](const glm::vec3& v) { return v.x > 0.0f || v.y > 0.0f || v.z > 0.0f; };
        return any(ambient) || any(zenith) || any(horizon) || any(ground);
    }
    bool active() const { return has_sun() || has_dome(); }

    float solid_angle() const {
        const float a = glm::radians(std::max(1e-3f, angle));
        return 6.283185307179586f * (1.0f - std::cos(a));
    }

    void bind(const Pipeline& p) const {
        const float a = glm::radians(std::max(1e-3f, angle));
        const glm::vec3 d = glm::length(dir) > 1e-6f ? glm::normalize(dir)
                                                     : glm::vec3(0.0f, 1.0f, 0.0f);
        const glm::vec3 u = glm::length(up) > 1e-6f ? glm::normalize(up)
                                                    : glm::vec3(0.0f, 1.0f, 0.0f);
        p.set("u_sky", ambient);
        p.set("u_sky_zenith", zenith);
        p.set("u_sky_horizon", horizon);
        p.set("u_sky_ground", ground);
        p.set("u_sky_up", u);
        p.set("u_sun_dir", d);
        p.set("u_sun_irr", irradiance);
        p.set("u_sun_cos_r", std::cos(a));
        p.set("u_sun_tan_r", std::tan(a));
        p.set("u_sun_inv_omega", 1.0f / solid_angle());
    }

    // MBG_DAYLIGHT=1. A late-morning sun coming in over the Cornell box's open
    // +z side from the upper left, so that it lands on the floor, the green wall
    // and the back wall and throws both boxes' shadows across the floor -- i.e.
    // so that the thing being added is visible rather than merely present.
    //
    // The sun is deliberately about twenty times the dome's total irradiance.
    // A sky bright enough to balance it is an overcast sky, and an overcast sky
    // is the case where this whole feature is invisible: every surface gets a
    // smooth hemisphere integral, the shadows go away, and the picture looks the
    // same whether the sun term is implemented or not. The interesting case is
    // the one where the direct term draws the image and the dome fills the
    // shadows, which is what these numbers are.
    //
    // The angular radius is 1.2 degrees, four and a half times the real sun's.
    // The real value gives a penumbra a few millimetres wide at Cornell's scale,
    // which is a hard edge in every sense that matters here and tests nothing
    // the emitter path does not already test. 1.2 puts the penumbra at a
    // centimetre or two: visibly soft, still obviously a sun, and wide enough
    // that the rasterized disc fraction has something to resolve.
    static SkyLight daylight() {
        SkyLight s;
        // Scaled so that the DEFAULT three-bounce solve lands in the tone
        // curve's usable range at exposure 1. A white box interreflects: put a
        // sun in one bright enough to read at one bounce and the third bounce
        // pushes every surface into Reinhard's shoulder, where the shadow the
        // ceiling throws across the tall box flattens out. Exposure would fix it
        // identically -- it multiplies the same linear radiance -- but a preset
        // that needs a second env var to look right is not a preset.
        s.zenith     = glm::vec3(0.050f, 0.094f, 0.198f);
        s.horizon    = glm::vec3(0.149f, 0.176f, 0.220f);
        s.ground     = glm::vec3(0.050f, 0.044f, 0.039f);
        s.irradiance = glm::vec3(4.62f, 4.24f, 3.63f);
        // Elevation 50 degrees, azimuth -30 from the opening. Chosen so that the
        // CEILING's leading edge cuts the beam: the shadow it throws lands
        // across the tall box and the back of the floor, which is the one thing
        // an interior lit through an aperture does that nothing else in this
        // scene was testing. A lower sun floods the whole floor evenly and a
        // higher one never reaches the back wall.
        s.dir        = glm::normalize(glm::vec3(-0.321f, 0.766f, 0.557f));
        s.angle      = 1.2f;
        return s;
    }
};

} // namespace mbg
