#include "animation.hpp"

#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>

namespace gfx {

static float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

static glm::vec3 lerp_vec3(const float* a, const float* b, float t) {
    glm::vec3 r;
    for (int i = 0; i < 3; ++i) r[i] = lerp(a[i], b[i], t);
    return r;
}

void AnimationClip::sample(float time, Skeleton& skeleton) const {
    int n = skeleton.joint_count();
    if (n == 0) return;
    if (duration <= 0.0f) return;

    // Accumulate animated TRS values per joint
    struct Accum {
        bool has_t = false, has_r = false, has_s = false;
        glm::vec3 t{0.0f};
        glm::quat r{1.0f, 0.0f, 0.0f, 0.0f};
        glm::vec3 s{1.0f};
    };
    std::vector<Accum> acc(n);

    for (const auto& ch : channels) {
        int ji = ch.joint_index;
        if (ji < 0 || ji >= n) continue;
        if (ch.sampler < 0 || ch.sampler >= static_cast<int>(samplers.size())) continue;

        const auto& sampler = samplers[ch.sampler];
        const auto& input = sampler.input;
        const auto& output = sampler.output;
        int nk = static_cast<int>(input.size());
        if (nk == 0 || output.empty()) continue;

        float t = glm::clamp(time, input.front(), input.back());

        // Find segment
        int idx = 0;
        for (int i = 0; i < nk - 1; ++i) {
            if (t >= input[i] && t < input[i + 1]) { idx = i; break; }
        }
        if (t >= input.back() && nk >= 2) idx = nk - 2;

        float t0 = input[idx], t1 = input[idx + 1];
        float frac = (t1 > t0) ? (t - t0) / (t1 - t0) : 0.0f;
        if (sampler.interpolation == 1) frac = 0.0f; // STEP

        int stride = sampler.output_stride;
        const float* v0 = output.data() + idx * stride;
        const float* v1 = output.data() + (idx + 1) * stride;

        if (ch.path == "translation") {
            acc[ji].has_t = true;
            acc[ji].t = lerp_vec3(v0, v1, frac);
        } else if (ch.path == "rotation") {
            acc[ji].has_r = true;
            glm::quat q0(v0[3], v0[0], v0[1], v0[2]);
            glm::quat q1(v1[3], v1[0], v1[1], v1[2]);
            acc[ji].r = glm::slerp(q0, q1, frac);
        } else if (ch.path == "scale") {
            acc[ji].has_s = true;
            acc[ji].s = lerp_vec3(v0, v1, frac);
        }
    }

    // Compose T * R * S for each animated joint; non-animated joints keep default
    for (int i = 0; i < n; ++i) {
        if (acc[i].has_t || acc[i].has_r || acc[i].has_s) {
            auto t = acc[i].has_t ? acc[i].t : skeleton.default_translation(i);
            auto r = acc[i].has_r ? acc[i].r : skeleton.default_rotation(i);
            auto s = acc[i].has_s ? acc[i].s : skeleton.default_scale(i);
            skeleton.joint_local_transform(i) =
                glm::translate(glm::mat4(1.0f), t) *
                glm::mat4_cast(r) *
                glm::scale(glm::mat4(1.0f), s);
        }
    }
}

} // namespace gfx
