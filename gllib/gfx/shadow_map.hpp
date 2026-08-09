#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

namespace gl {
class Program;
} // namespace gl

namespace gfx {

class Mesh;

class ShadowMap {
public:
    explicit ShadowMap(int size = 2048);
    ~ShadowMap();

    ShadowMap(const ShadowMap&) = delete;
    ShadowMap& operator=(const ShadowMap&) = delete;

    ShadowMap(ShadowMap&&) noexcept;
    ShadowMap& operator=(ShadowMap&&) noexcept;

    bool valid() const { return handle_ != 0; }

    void begin();
    void end();

    void render_mesh(const Mesh& mesh, const glm::mat4& mvp,
                     GLuint bone_ssbo = 0);

    void bind(int unit) const;

    GLuint handle() const { return handle_; }
    int size() const { return size_; }

private:
    GLuint handle_ = 0;
    GLuint fbo_ = 0;
    int size_ = 2048;
};

glm::mat4 compute_light_vp(const glm::mat4& camera_vp,
                           const glm::vec3& light_dir,
                           float scene_radius,
                           float cascade_distance);

} // namespace gfx
