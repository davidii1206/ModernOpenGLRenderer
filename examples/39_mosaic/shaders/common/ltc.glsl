#ifndef MOSAIC_LTC_GLSL
#define MOSAIC_LTC_GLSL

// Analytic area lights, shared by S2 (surfels) and the pixel composite.
//
// For a DIFFUSE receiver, "linearly transformed cosines" degenerates to the
// identity transform: the integral of a uniform-radiance polygon against a
// clamped cosine lobe is the classic Lambert/Arvo form factor, which is exact
// and needs no fitted tables at all. Diffuse is most of this image, so it goes
// in first and unconditionally; the fitted M^-1 tables are only needed for the
// glossy lobe, which is why the glossy path can slip without blocking anything.
//
// Mirrors mosaic::GpuEmitter (direct.hpp):
//   p0 = centre.xyz, area   p1 = half_u.xyz, radiance.r
//   p2 = half_v.xyz, rad.g  p3 = normal.xyz, radiance.b

struct Emitter {
    vec4 p0, p1, p2, p3;
};

vec3 emitter_center(Emitter e)   { return e.p0.xyz; }
float emitter_area(Emitter e)    { return e.p0.w; }
vec3 emitter_normal(Emitter e)   { return e.p3.xyz; }
vec3 emitter_radiance(Emitter e) { return vec3(e.p1.w, e.p2.w, e.p3.w); }

// Corner k of the rectangle. The host guarantees cross(u, v) == normal, and
// these wind counter-clockwise AS SEEN FROM THE EMITTING SIDE (the -normal
// half-space) -- which is the order the signed form factor below needs to come
// out positive for a receiver the light actually illuminates.
vec3 emitter_corner(Emitter e, int k) {
    vec3 u = e.p1.xyz, v = e.p2.xyz;
    if (k == 0) return e.p0.xyz - u - v;
    if (k == 1) return e.p0.xyz - u + v;
    if (k == 2) return e.p0.xyz + u + v;
    return e.p0.xyz + u - v;
}

// Cosine-weighted solid angle of the polygon as seen from P with normal N,
// i.e. E / L. Range [0, pi]: pi when the polygon fills the hemisphere.
//
// Sum over edges of the angle between the two vertex directions, weighted by
// how the edge's great-circle normal aligns with N. Clipping is handled by
// clamping the result rather than by splitting the polygon against the horizon:
// the error is confined to lights that straddle the tangent plane, where it
// shows up as a slightly soft terminator rather than as light leaking through.
float polygon_form_factor(vec3 P, vec3 N, Emitter e) {
    vec3 L[4];
    for (int i = 0; i < 4; ++i) {
        vec3 d = emitter_corner(e, i) - P;
        float len = length(d);
        if (len < 1e-8) return 0.0;
        L[i] = d / len;
    }

    float sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        vec3 a = L[i];
        vec3 b = L[(i + 1) & 3];
        float c = clamp(dot(a, b), -1.0, 1.0);
        vec3 cr = cross(a, b);
        float crl = length(cr);
        if (crl < 1e-8) continue;
        sum += acos(c) * dot(cr / crl, N);
    }
    return max(0.5 * sum, 0.0);
}

// Irradiance / pi from one emitter, which is the same convention the SH cache
// uses (sh_eval_irradiance also returns E/pi), so the two are directly additive
// and a Lambertian surface's outgoing radiance is albedo times their sum.
//
// One-sided: a rectangle only emits from its front face. Without the test the
// Cornell ceiling panel lights the far side of the ceiling as brightly as the
// room.
vec3 emitter_irradiance(vec3 P, vec3 N, Emitter e) {
    vec3 to = emitter_center(e) - P;
    if (dot(to, emitter_normal(e)) >= 0.0) return vec3(0.0);
    float ff = polygon_form_factor(P, N, e);
    if (ff <= 0.0) return vec3(0.0);
    return emitter_radiance(e) * (ff * (1.0 / 3.14159265));
}

#endif // MOSAIC_LTC_GLSL
