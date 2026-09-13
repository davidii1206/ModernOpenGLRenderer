#ifndef MBG_SKY_GLSL
#define MBG_SKY_GLSL

// ---------------------------------------------------------------------------
// The sky dome and the sun.
//
// THE SKY IS ALREADY IMPLEMENTED. That is the whole point of it in this
// renderer: a secondary camera's hemisphere is a visibility buffer, and a texel
// that no triangle reached is a texel that sees the sky. There is no extra pass,
// no cube map lookup chain and no special case -- raster.comp's quadrature loop
// already visits every empty texel, it just used to add a constant there. Making
// that constant a function of the direction is the entire sky feature, and the
// cosine-weighted solid angle it is multiplied by is the same table every other
// term uses. Sky occlusion therefore comes out exactly as correct as the
// rasterizer is, which is the same statement as for every other bounce.
//
// THE SUN CANNOT WORK THAT WAY, and the reason is the same one that made
// Cornell's panel analytic (common/emitter.glsl). A disc of one degree covers
// 3e-5 of the hemisphere; at a 16x16 target that is a two-thousandth of a texel,
// so a quadrature would find it in one camera out of two thousand and miss it in
// the rest. The answer is the same split used for the panel:
//
//   magnitude    analytic and exact -- L * Omega * cos(theta), below
//   visibility   rasterized, off a frustum fitted to the DISC
//                (lv_setup_dir / lv_mass_disc in common/lightview.glsl)
//
// and it is cheaper than the panel's, because a light at infinity has nothing
// behind it: there is no emitter rasterization to divide by. A texel of the disc
// is lit exactly when nothing was written into it.
//
// The dome deliberately does NOT contain the sun disc. If it did, the analytic
// term would be counted twice in the one camera in two thousand that happens to
// land a texel on it -- as a 2000x spike, since that texel's weight is sized for
// a whole texel of sky.
//
// The gradient itself is not a physical sky model (no Preetham, no Hosek): this
// example measures transport, and all the transport needs from a dome is that
// its radiance vary with direction. Three colours and a horizon falloff give
// that, and stay adjustable enough to be dialled to a scene by eye.
// ---------------------------------------------------------------------------

uniform vec3  u_sky;            // constant floor added in every direction
uniform vec3  u_sky_zenith;
uniform vec3  u_sky_horizon;
uniform vec3  u_sky_ground;     // below the horizon: bounce off the ground plane
uniform vec3  u_sky_up;         // world up; glTF is Y-up, but nothing here assumes it
uniform vec3  u_sun_dir;        // unit, pointing TOWARD the sun
uniform vec3  u_sun_irr;        // irradiance on a surface facing the sun, = L * Omega
uniform float u_sun_cos_r;      // cos of the sun's angular radius
uniform float u_sun_tan_r;      // tan of it
uniform float u_sun_inv_omega;  // 1 / Omega, for recovering the disc's radiance

// Radiance arriving from `d` with nothing in the way. `d` need not be unit.
//
// sqrt, not a linear ramp: it puts most of the change close to the horizon,
// which is where a real sky's is, and it costs nothing.
vec3 mbg_sky_dome(vec3 d) {
    float t = dot(normalize(d), u_sky_up);
    return u_sky + (t >= 0.0 ? mix(u_sky_horizon, u_sky_zenith, sqrt(t))
                             : mix(u_sky_horizon, u_sky_ground,  sqrt(-t)));
}

// The sun's UNSHADOWED irradiance on a surface with normal `nrm`.
//
// E = L * Omega * cos(theta), with L * Omega supplied as one number because that
// is the quantity a user actually wants to set: "how bright is a surface facing
// the sun". Splitting it would make the angular radius change the exposure,
// which is not what a softness control should do.
//
// The cosine is clamped rather than integrated over the part of the disc above
// the horizon. At a degree of angular radius that approximation is wrong only
// within a degree of the terminator, where the term it is wrong about is already
// within a degree of zero -- so the absolute error is O(alpha^2) ~ 3e-4 of the
// sun's own irradiance, against an emitter formula that is exact because a
// Cornell panel subtends thirty degrees and there the same shortcut would not be
// available.
vec3 mbg_sun_unshadowed(vec3 nrm) {
    return u_sun_irr * max(0.0, dot(nrm, u_sun_dir));
}

// The disc's own radiance, for looking straight at it. Only the display pass
// wants this: every shading path uses the irradiance above.
vec3 mbg_sun_radiance() { return u_sun_irr * u_sun_inv_omega; }

// What a camera ray that hit nothing should show: the dome, plus the disc when
// the ray is inside it.
vec3 mbg_sky_lookup(vec3 d) {
    vec3 w = normalize(d);
    vec3 c = mbg_sky_dome(w);
    if (dot(w, u_sun_dir) >= u_sun_cos_r) c += mbg_sun_radiance();
    return c;
}

#endif
