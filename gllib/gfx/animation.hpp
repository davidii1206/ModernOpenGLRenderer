#pragma once

#include "skeleton.hpp"

#include <string>
#include <vector>

namespace gfx {

struct AnimationSampler {
    std::vector<float> input;         // timestamps
    std::vector<float> output;        // keyframe data
    int interpolation = 0;            // TINYGLTF_ANIMATION_INTERPOLATION_LINEAR
    int output_stride = 3;            // 3 (translation/scale) or 4 (rotation)
};

struct AnimationChannel {
    int joint_index = -1;
    int sampler = -1;
    std::string path;                 // "translation", "rotation", "scale"
};

class AnimationClip {
public:
    AnimationClip() = default;
    ~AnimationClip() = default;
    AnimationClip(AnimationClip&&) noexcept = default;
    AnimationClip& operator=(AnimationClip&&) noexcept = default;

    std::string name;
    float duration = 0.0f;
    std::vector<AnimationSampler> samplers;
    std::vector<AnimationChannel> channels;

    void sample(float time, Skeleton& skeleton) const;
};

} // namespace gfx
