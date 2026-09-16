#include "scene.hpp"

#include <gllib/log.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace mbg {

// --- Triangle extraction ----------------------------------------------------
//
// Byte-for-byte the same extraction as example 40's surfels.cpp, so the two
// examples are demonstrably lighting the identical geometry with the identical
// materials -- which is the only way their images are comparable.

std::vector<Tri> extract_triangles(const gfx::Model& model) {
    std::vector<Tri> out;

    for (std::size_t mi = 0; mi < model.mesh_count(); ++mi) {
        const gfx::Mesh& mesh = model.mesh(mi);
        const glm::mat4  xf   = model.mesh_transform(mi);
        const glm::mat3  nxf  = glm::transpose(glm::inverse(glm::mat3(xf)));

        glm::vec3 albedo(1.0f), emission(0.0f);
        bool two_sided = false;
        const int mat = model.mesh_material(mi);
        if (mat >= 0 && std::size_t(mat) < model.material_count()) {
            const gfx::ModelMaterialInfo& m = model.material_info(std::size_t(mat));
            albedo = glm::vec3(m.base_color_factor[0], m.base_color_factor[1],
                               m.base_color_factor[2]);
            // emissive_factor already carries KHR_materials_emissive_strength;
            // CornellBoxOriginal.glb authors its ceiling panel at strength 17,
            // so dropping the extension renders it 17x too dim.
            emission = glm::vec3(m.emissive_factor[0], m.emissive_factor[1],
                                 m.emissive_factor[2]);
            two_sided = m.double_sided;
        }

        const std::vector<gfx::Vertex>& vs = mesh.vertices();
        const std::vector<unsigned int>& is = mesh.indices();

        auto emit = [&](const gfx::Vertex& a, const gfx::Vertex& b, const gfx::Vertex& c) {
            Tri t;
            const glm::vec3 lp[3] = {
                glm::vec3(a.position[0], a.position[1], a.position[2]),
                glm::vec3(b.position[0], b.position[1], b.position[2]),
                glm::vec3(c.position[0], c.position[1], c.position[2]),
            };
            for (int k = 0; k < 3; ++k) t.p[k] = glm::vec3(xf * glm::vec4(lp[k], 1.0f));

            const glm::vec3 cr = glm::cross(t.p[1] - t.p[0], t.p[2] - t.p[0]);
            const float len = glm::length(cr);
            if (len < 1e-12f) return;                 // degenerate
            t.area = 0.5f * len;

            // Geometric normal, oriented to agree with the shading normal so a
            // clockwise-wound triangle does not end up emitting backwards.
            glm::vec3 gn = cr / len;
            const glm::vec3 sn = nxf * glm::vec3(a.normal[0], a.normal[1], a.normal[2]);
            if (glm::dot(gn, sn) < 0.0f) gn = -gn;
            t.n = gn;

            t.albedo = albedo;
            t.emission = emission;
            t.double_sided = two_sided;
            out.push_back(t);
        };

        if (is.empty()) {
            for (std::size_t v = 0; v + 2 < vs.size(); v += 3) emit(vs[v], vs[v + 1], vs[v + 2]);
        } else {
            for (std::size_t k = 0; k + 2 < is.size(); k += 3)
                emit(vs[is[k]], vs[is[k + 1]], vs[is[k + 2]]);
        }
    }

    // --- Spatial order, and it is what makes the cluster levels work ---------
    //
    // Clusters are runs of consecutive triangles, so their boxes are only as
    // tight as the order they arrive in. Mesh order gives decent 64-triangle
    // clusters -- a mesh's triangles are locally coherent -- and USELESS groups
    // above them: 64 consecutive clusters of Sponza is 4096 triangles spanning
    // the whole building, so the coarse box contains everything and culls
    // nothing. Measured: adding the coarse level on mesh order was worth 1%.
    //
    // Sorting by Morton code makes both levels compact, because a run of
    // consecutive codes is a run of nearby triangles at every scale.
    //
    // AND IT IS NOT UNIFORMLY BETTER, which is worth knowing before trusting
    // it. Measured at a 32x32 grid, one bounce:
    //
    //   Cornell+bunny  direct      5806 ms -> 3134 ms   1.85x
    //   Cornell+bunny  hemisphere  2860 ms -> 2413 ms   1.19x
    //   Sponza         hemisphere 14948 ms -> 17192 ms  0.87x   (three runs, +-1%)
    //
    // The bunny is one dense organic mesh, where the model's own order carries
    // no useful structure and Morton supplies it. Sponza is architecture, where
    // the artist already put each column's triangles together -- tighter than a
    // Morton run that straddles a seam, which is the known weakness of the
    // curve. So this helps a mesh and hurts a building.
    //
    // Done HERE rather than in Scene::build so the host's triangle vector and
    // the GPU's agree on indices -- the oracle gate compares them directly.
    // MBG_MORTON=0 keeps the model's own order, which is not always worse --
    // see the measurement in the header comment above.
    const char* morton_env = getenv("MBG_MORTON");
    const bool morton = !(morton_env && atoi(morton_env) == 0);
    if (morton && out.size() > 1) {
        glm::vec3 lo(1e30f), hi(-1e30f);
        for (const Tri& t : out)
            for (int k = 0; k < 3; ++k) { lo = glm::min(lo, t.p[k]); hi = glm::max(hi, t.p[k]); }
        const glm::vec3 inv = 1.0f / glm::max(hi - lo, glm::vec3(1e-6f));

        // 10 bits per axis interleaved: one 32-bit key, no big-integer work.
        auto spread = [](uint32_t v) {
            v &= 0x3FFu;
            v = (v | (v << 16)) & 0x030000FFu;
            v = (v | (v <<  8)) & 0x0300F00Fu;
            v = (v | (v <<  4)) & 0x030C30C3u;
            v = (v | (v <<  2)) & 0x09249249u;
            return v;
        };
        std::vector<std::pair<uint32_t, uint32_t>> order(out.size());
        for (std::size_t i = 0; i < out.size(); ++i) {
            const glm::vec3 c = (out[i].p[0] + out[i].p[1] + out[i].p[2]) / 3.0f;
            const glm::vec3 u = glm::clamp((c - lo) * inv, glm::vec3(0.0f), glm::vec3(1.0f));
            order[i] = {spread(uint32_t(u.x * 1023.0f)) |
                        (spread(uint32_t(u.y * 1023.0f)) << 1) |
                        (spread(uint32_t(u.z * 1023.0f)) << 2),
                        uint32_t(i)};
        }
        std::sort(order.begin(), order.end());
        std::vector<Tri> sorted(out.size());
        for (std::size_t i = 0; i < out.size(); ++i) sorted[i] = out[order[i].second];
        out.swap(sorted);
    }
    return out;
}

double total_area(const std::vector<Tri>& tris) {
    double a = 0.0;
    for (const Tri& t : tris) a += double(t.area);
    return a;
}

// --- Scene ------------------------------------------------------------------

bool Scene::build(const std::vector<Tri>& tris, uint32_t super_size) {
    if (tris.size() > kMaxTris) {
        gllib::logf(gllib::LogLevel::error,
                    "%zu triangles exceeds the %u the visibility key allows",
                    tris.size(), kMaxTris);
        return false;
    }

    // Cluster bounds first: a run of kClusterSize consecutive triangles and the
    // box around them. Built here rather than on the fly because it is static
    // scene data, and read by every light view that wants to skip most of it.
    std::vector<GpuCluster> clusters;
    for (std::size_t base = 0; base < tris.size(); base += kClusterSize) {
        const std::size_t n = std::min<std::size_t>(kClusterSize, tris.size() - base);
        glm::vec3 lo(1e30f), hi(-1e30f);
        for (std::size_t i = 0; i < n; ++i)
            for (int k = 0; k < 3; ++k) {
                lo = glm::min(lo, tris[base + i].p[k]);
                hi = glm::max(hi, tris[base + i].p[k]);
            }
        clusters.push_back({glm::vec4(lo, float(base)), glm::vec4(hi, float(n))});
    }
    cluster_count_ = uint32_t(clusters.size());
    clusters_.data(clusters.data(), clusters.size() * sizeof(GpuCluster));

    // The coarse level: a box over each run of kGroupSize clusters. Without it
    // every texel tests every cluster, and 256 texels sharing one origin each
    // rediscover the same scene independently.
    std::vector<GpuCluster> groups;
    for (std::size_t base = 0; base < clusters.size(); base += kGroupSize) {
        const std::size_t n = std::min<std::size_t>(kGroupSize, clusters.size() - base);
        glm::vec3 lo(1e30f), hi(-1e30f);
        for (std::size_t i = 0; i < n; ++i) {
            lo = glm::min(lo, glm::vec3(clusters[base + i].lo));
            hi = glm::max(hi, glm::vec3(clusters[base + i].hi));
        }
        groups.push_back({glm::vec4(lo, float(base)), glm::vec4(hi, float(n))});
    }
    group_count_ = uint32_t(groups.size());
    groups_.data(groups.data(), groups.size() * sizeof(GpuCluster));

    // The third level, over runs of groups. Same shape as the two below it,
    // because the thing that was missing was a level and not a different kind of
    // structure: without it the group list is scanned in full by every
    // camera-thread, which is O(scene) and 68% of Bistro's sweep (finding 46).
    //
    // super_size 0 means ONE super-group over everything, which is the old
    // traversal plus a single box test -- the A/B baseline, built from the same
    // code so the comparison cannot drift.
    const std::size_t sstep = super_size ? std::size_t(super_size) : groups.size();
    std::vector<GpuCluster> supers;
    for (std::size_t base = 0; base < groups.size(); base += std::max<std::size_t>(1, sstep)) {
        const std::size_t n = std::min(std::max<std::size_t>(1, sstep), groups.size() - base);
        glm::vec3 lo(1e30f), hi(-1e30f);
        for (std::size_t i = 0; i < n; ++i) {
            lo = glm::min(lo, glm::vec3(groups[base + i].lo));
            hi = glm::max(hi, glm::vec3(groups[base + i].hi));
        }
        supers.push_back({glm::vec4(lo, float(base)), glm::vec4(hi, float(n))});
    }
    super_count_ = uint32_t(supers.size());
    supers_.data(supers.data(), supers.size() * sizeof(GpuCluster));

    std::vector<GpuTriGeom> gpu(tris.size());
    std::vector<GpuTriShade> shade(tris.size());
    std::vector<uint32_t> emitters;
    bounds_ = Bounds{};
    area_ = 0.0;
    emissive_ = 0;
    for (std::size_t i = 0; i < tris.size(); ++i) {
        const Tri& t = tris[i];
        GpuTriGeom& g = gpu[i];
        GpuTriShade& sh = shade[i];
        g.p0 = glm::vec4(t.p[0], 0.0f);
        g.p1 = glm::vec4(t.p[1], 0.0f);
        g.p2 = glm::vec4(t.p[2], 0.0f);
        g.n  = glm::vec4(t.n, t.double_sided ? 1.0f : 0.0f);
        sh.albedo = glm::vec4(t.albedo, 0.0f);
        // emission.w flags a triangle as an ANALYTIC emitter: the hemisphere
        // resolve then skips its L_e, because raster.comp adds the same energy
        // exactly instead of quadrature-sampled. Flagging it in the triangle
        // rather than only in the emitter list is what makes the two paths
        // impossible to double-count -- one flag gates both.
        const bool emits = glm::dot(t.emission, glm::vec3(1.0f)) > 0.0f;
        sh.emission = glm::vec4(t.emission, emits ? 1.0f : 0.0f);
        if (emits) emitters.push_back(uint32_t(i));
        for (int k = 0; k < 3; ++k) bounds_.add(t.p[k]);
        area_ += double(t.area);
        if (emits) ++emissive_;
    }

    count_ = uint32_t(gpu.size());
    buf_.data(gpu.data(), gpu.size() * sizeof(GpuTriGeom));
    shade_.data(shade.data(), shade.size() * sizeof(GpuTriShade));
    emitter_count_ = uint32_t(emitters.size());
    // Never leave the binding empty: a shader that reads an unbound SSBO is
    // undefined, and u_emitters == 0 is a value the kernel has to be given
    // rather than a state it can infer.
    if (emitters.empty()) emitters.push_back(0u);
    emit_.data(emitters.data(), emitters.size() * sizeof(uint32_t));
    return count_ > 0;
}

// --- Quadrature -------------------------------------------------------------

glm::vec3 hemi_oct_dir(glm::vec2 e) {
    const glm::vec2 t{(e.x + e.y) * 0.5f, (e.x - e.y) * 0.5f};
    return glm::vec3(t.x, t.y, 1.0f - std::abs(t.x) - std::abs(t.y));
}

void Quadrature::build(uint32_t r, uint32_t subsamples) {
    res = r;
    const uint32_t n = r * r;
    const uint32_t S = std::max(1u, subsamples);
    texels.assign(n, glm::vec4(0.0f));

    const double de = 2.0 / double(r);          // texel edge in square coordinates
    const double ds = de / double(S);           // subsample edge
    const double cell = ds * ds;

    sum_omega = 0.0;
    sum_cos = 0.0;

    for (uint32_t ty = 0; ty < r; ++ty) {
        for (uint32_t tx = 0; tx < r; ++tx) {
            double w_omega = 0.0, w_cos = 0.0;
            for (uint32_t sy = 0; sy < S; ++sy) {
                for (uint32_t sx = 0; sx < S; ++sx) {
                    const double ex = -1.0 + (double(tx) + (double(sx) + 0.5) / double(S)) * de;
                    const double ey = -1.0 + (double(ty) + (double(sy) + 0.5) / double(S)) * de;
                    const glm::vec3 w = hemi_oct_dir(glm::vec2(float(ex), float(ey)));
                    const double len = std::sqrt(double(w.x) * w.x + double(w.y) * w.y +
                                                 double(w.z) * w.z);
                    // dOmega = (1/2) |w|^-3 de, and the cosine against the
                    // camera's own normal (+Z) adds w.z/|w|.
                    const double inv = 1.0 / (len * len * len);
                    w_omega += 0.5 * inv * cell;
                    w_cos   += 0.5 * inv * (double(w.z) / len) * cell;
                }
            }
            // The direction is the texel CENTRE, not the centroid of the
            // (slightly asymmetric) spherical patch. That difference is the
            // quadrature's remaining error and it is what more texels buys.
            const glm::vec3 c = glm::normalize(hemi_oct_dir(glm::vec2(
                float(-1.0 + (double(tx) + 0.5) * de),
                float(-1.0 + (double(ty) + 0.5) * de))));
            texels[ty * r + tx] = glm::vec4(c, float(w_cos));
            sum_omega += w_omega;
            sum_cos += w_cos;
        }
    }

    // Normalize to PI exactly. The residual before this is ~1e-7 relative at the
    // default subsampling, so it changes nothing visible -- but it makes "a
    // camera fully enclosed by radiance L reads exactly PI*L" an identity rather
    // than an approximation, and that identity is gate 1.
    const double k = sum_cos > 0.0 ? 3.14159265358979323846 / sum_cos : 1.0;
    for (glm::vec4& t : texels) t.w = float(double(t.w) * k);

    buf.data(texels.data(), texels.size() * sizeof(glm::vec4));
}

} // namespace mbg
