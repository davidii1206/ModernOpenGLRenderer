#ifndef MOSAIC_CLIPMAP_GLSL
#define MOSAIC_CLIPMAP_GLSL

// Cascade addressing shared by every S1 consumer.
//
// Three camera-centred 64^3 cascades, cell size growing by CASCADE_RATIO per
// level, rebuilt from scratch each frame. Cell (0,0,0)'s min corner is snapped
// to a multiple of the cell size so cells do not swim as the camera moves;
// without the snap, every cell boundary jitters sub-cell each frame and the
// cluster attributes flicker.
//
// Note that cluster IDs are NOT stable across frames — the whole structure is
// rebuilt — so nothing persistent may store one. In particular the S4 reservoir
// stores a quantized world position, not a cluster id.

const int CLIP_RES = 64;
const int CLIP_CELLS = CLIP_RES * CLIP_RES * CLIP_RES;

int clip_cell_index(ivec3 c) {
    return c.x + CLIP_RES * (c.y + CLIP_RES * c.z);
}

ivec3 clip_cell_coord(vec3 p, vec3 origin, float inv_cell) {
    return ivec3(floor((p - origin) * inv_cell));
}

bool clip_inside(ivec3 c) {
    return all(greaterThanEqual(c, ivec3(0))) && all(lessThan(c, ivec3(CLIP_RES)));
}

// Selects the finest cascade that contains p. Returns -1 when p falls outside
// every cascade (beyond the outermost coverage).
int clip_select(vec3 p, vec3 origins[3], float inv_cells[3]) {
    for (int l = 0; l < 3; ++l) {
        if (clip_inside(clip_cell_coord(p, origins[l], inv_cells[l]))) return l;
    }
    return -1;
}

// --- RGB9E5 ----------------------------------------------------------------
// Matches pack_rgb9e5 in surfel_bake.cpp and GL_RGB9_E5.

vec3 unpack_rgb9e5(uint v) {
    float e = exp2(float(int(v >> 27u) - 15 - 9));
    return vec3(float(v & 0x1FFu), float((v >> 9u) & 0x1FFu), float((v >> 18u) & 0x1FFu)) * e;
}

uint pack_rgb9e5(vec3 c) {
    c = clamp(c, vec3(0.0), vec3(65408.0));
    float maxc = max(c.r, max(c.g, c.b));
    if (maxc < 1e-9) return 0u;
    int e = max(-16, int(floor(log2(maxc)))) + 1;
    float d = exp2(float(e - 15 - 9));
    if (int(round(maxc / d)) == 512) { e += 1; d = exp2(float(e - 15 - 9)); }
    uvec3 m = uvec3(clamp(round(c / d), vec3(0.0), vec3(511.0)));
    return m.r | (m.g << 9u) | (m.b << 18u) | (uint(clamp(e + 15, 0, 31)) << 27u);
}

#endif
