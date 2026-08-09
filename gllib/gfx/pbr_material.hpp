#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <cstddef>

namespace gl {
class Program;
} // namespace gl

namespace gfx {

class IBLProbe;
class ShadowMap;
class Model;
struct ModelMaterialInfo;
class Mesh;
class Cubemap;

class PBRMaterial {
public:
    PBRMaterial();
    ~PBRMaterial();

    PBRMaterial(const PBRMaterial&) = delete;
    PBRMaterial& operator=(const PBRMaterial&) = delete;

    PBRMaterial(PBRMaterial&&) noexcept;
    PBRMaterial& operator=(PBRMaterial&&) noexcept;

    bool valid() const { return valid_; }

    void begin(const glm::mat4& view, const glm::mat4& proj, const glm::vec3& view_pos);
    void set_model_matrix(const glm::mat4& model);
    void set_ibl(const IBLProbe& ibl);
    void set_shadow(const ShadowMap& shadow, const glm::mat4& light_vp);
    void set_material(const ModelMaterialInfo& mat, const Model& model);
    void set_skin(GLuint bone_ssbo);
    void set_ambient_hemi(const glm::vec3& top, const glm::vec3& bottom, float intensity);
    void draw(const Mesh& mesh);
    void end();

private:
    gl::Program* prog_ = nullptr;
    bool valid_ = false;
};

} // namespace gfx
