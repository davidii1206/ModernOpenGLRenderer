#ifndef MOSAIC_VISIBILITY_GLSL
#define MOSAIC_VISIBILITY_GLSL

// T2 of the spec's visibility ladder: a cone march through the occupancy
// clipmap.
//
// The occupancy cascades hold binary coverage at kOccRes per cascade, mip-mapped
// with a BOX filter, so mip m of a texel is the fraction of that region which is
// occupied. Marching a cone and sampling the mip whose voxel matches the cone's
// current radius is then a coverage integral, not a point test: it produces a
// soft, wide-source shadow without any sampling noise, which is exactly what an
// area light wants.
//
// This deliberately carries no radiance -- keeping the occupancy field binary is
// what keeps a voxel-GI leak class out of the pipeline entirely.
//
// Requires from the host:
//   sampler3D u_occupancy[3], u_occ_origins[3], u_occ_inv_cells[3],
//   u_occ_cascades, u_occ_mips

float occ_sample(vec3 p, float lod) {
    for (int l = 0; l < u_occ_cascades; ++l) {
        vec3 uvw = (p - u_occ_origins[l]) * u_occ_inv_cells[l] * (1.0 / float(OCC_RES));
        if (all(greaterThanEqual(uvw, vec3(0.0))) && all(lessThan(uvw, vec3(1.0))))
            return textureLod(u_occupancy[l], uvw, clamp(lod, 0.0, float(u_occ_mips - 1))).r;
    }
    return 0.0;   // outside every cascade: nothing known to occlude
}

// Marches from P toward a source `dist` away, with a cone whose half-angle is
// set by `spread` (tan of the half-angle; roughly source_radius / distance).
// Returns visibility in [0, 1].
float cone_visibility(vec3 P, vec3 N, vec3 dir, float dist, float spread, int steps) {
    if (steps <= 0) return 1.0;

    // Start off the surface, or the receiver's own voxel shadows it. One and a
    // half cells along the normal is the bias example 33 needed for the same
    // reason, scaled to the finest occupancy cell here.
    float cell0 = 1.0 / u_occ_inv_cells[0];
    float t = 1.5 * cell0;
    // Stop a source-RADIUS short, not a cell short.
    //
    // The cone widens to the source's own half-extent by the time it arrives, so
    // for a light set into a ceiling the last samples always straddle the plane
    // the light sits in and read it as an occluder. Whether a given sample lands
    // inside that layer depends on where the geometric progression happens to
    // put it, which varies with distance -- and that drew concentric rings
    // around the emitter on every wall, sharper at higher step counts because
    // more samples landed near the boundary. Ending the march clear of the
    // source's own neighbourhood removes the whole class.
    float src_r = spread * dist;
    float t_end = max(dist - src_r - 1.5 * cell0, t * 1.001);

    // The march must REACH the source inside its step budget.
    //
    // Pure cone stepping grows t by (1 + spread) per step, which for a small
    // source is far too slow: at spread 0.12 sixteen steps cover 6x the start
    // radius, well short of a room. The march then stopped mid-air with partial
    // occlusion accumulated, and since where it stopped depends on the distance
    // to the light, the truncation drew concentric rings around the emitter on
    // every wall. Taking the geometric ratio that lands exactly on t_end in
    // `steps` fixes both the rings and the cost, which is now bounded.
    float growth = max(1.0 + spread, pow(t_end / t, 1.0 / float(steps)));

    float occ = 0.0;
    for (int i = 0; i < steps && t < t_end; ++i) {
        float step_len = t * (growth - 1.0);
        // Footprint matched to whichever is larger, the cone or the step: a step
        // bigger than the cone would otherwise tunnel straight through a wall
        // thinner than the gap it jumps.
        float radius = max(max(spread * t, 0.5 * step_len), 0.5 * cell0);
        vec3 p = P + N * (0.5 * cell0) + dir * t;

        // Mip whose voxel edge matches the footprint diameter. textureLod
        // filters between mips, so this stays continuous as the cone widens.
        float lod = log2(max(2.0 * radius / cell0, 1.0));
        float s = occ_sample(p, lod);

        // Front-to-back alpha compositing, the standard cone-trace
        // accumulation: an occluder found early hides what is behind it instead
        // of adding to it, so a doubled wall does not read as twice as opaque.
        occ += (1.0 - occ) * s;
        if (occ > 0.995) break;

        t += step_len;
    }
    return clamp(1.0 - occ, 0.0, 1.0);
}

#endif // MOSAIC_VISIBILITY_GLSL
