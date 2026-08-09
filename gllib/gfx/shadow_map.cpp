#include "shadow_map.hpp"
#include "mesh.hpp"
#include "../gl/shader.hpp"
#include "../gl/program.hpp"
#include "../gl/texture.hpp"
#include "../gl/framebuffer.hpp"
#include "../gl/state.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <array>
#include <limits>
#include <cmath>

static const char* depth_vs = R"(
#version 430 core
layout(location = 0) in vec3 a_pos;
layout(location = 4) in uvec4 a_bone_indices;
layout(location = 5) in vec4 a_bone_weights;

layout(std430, binding = 4) readonly buffer BoneMatrices {
    mat4 u_bone_matrices[];
};

uniform mat4 u_mvp;
uniform bool u_has_skin;

void main() {
    vec4 pos = vec4(a_pos, 1.0);
    if (u_has_skin) {
        mat4 skin = mat4(0);
        for (int i = 0; i < 4; ++i) {
            skin += a_bone_weights[i] * u_bone_matrices[a_bone_indices[i]];
        }
        pos = skin * pos;
    }
    gl_Position = u_mvp * pos;
}
)";

static const char* depth_fs = R"(
#version 430 core
void main() {}
)";

static gl::Program* s_depth_prog = nullptr;

static gl::Program& get_depth_prog() {
    if (!s_depth_prog) {
        s_depth_prog = new gl::Program;
        gl::Shader vs(gl::ShaderType::vertex, depth_vs);
        gl::Shader fs(gl::ShaderType::fragment, depth_fs);
        if (vs.compiled() && fs.compiled()) {
            s_depth_prog->attach(vs);
            s_depth_prog->attach(fs);
            s_depth_prog->link();
        }
    }
    return *s_depth_prog;
}

gfx::ShadowMap::ShadowMap(int size)
    : size_(size)
{
    glCreateTextures(GL_TEXTURE_2D, 1, &handle_);
    glTextureStorage2D(handle_, 1, GL_DEPTH_COMPONENT24, size, size);
    glTextureParameteri(handle_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(handle_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(handle_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTextureParameteri(handle_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    const float border[] = {1.0f, 1.0f, 1.0f, 1.0f};
    glTextureParameterfv(handle_, GL_TEXTURE_BORDER_COLOR, border);
    // Manual PCF in shader — no hardware compare mode

    glCreateFramebuffers(1, &fbo_);
    glNamedFramebufferTexture(fbo_, GL_DEPTH_ATTACHMENT, handle_, 0);
    glNamedFramebufferDrawBuffer(fbo_, GL_NONE);
    glNamedFramebufferReadBuffer(fbo_, GL_NONE);
    GLenum status = glCheckNamedFramebufferStatus(fbo_, GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        glDeleteFramebuffers(1, &fbo_);
        glDeleteTextures(1, &handle_);
        fbo_ = 0;
        handle_ = 0;
    }
}

gfx::ShadowMap::~ShadowMap() {
    if (fbo_) glDeleteFramebuffers(1, &fbo_);
    if (handle_) glDeleteTextures(1, &handle_);
}

gfx::ShadowMap::ShadowMap(ShadowMap&& other) noexcept
    : handle_(other.handle_), fbo_(other.fbo_), size_(other.size_)
{
    other.handle_ = 0;
    other.fbo_ = 0;
}

gfx::ShadowMap& gfx::ShadowMap::operator=(ShadowMap&& other) noexcept {
    if (this != &other) {
        if (fbo_) glDeleteFramebuffers(1, &fbo_);
        if (handle_) glDeleteTextures(1, &handle_);
        handle_ = other.handle_;
        fbo_ = other.fbo_;
        size_ = other.size_;
        other.handle_ = 0;
        other.fbo_ = 0;
    }
    return *this;
}

void gfx::ShadowMap::begin() {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glViewport(0, 0, size_, size_);
    gl::enable(GL_DEPTH_TEST);
    gl::depth_func(GL_LESS);
    gl::depth_mask(GL_TRUE);
    gl::disable(GL_CULL_FACE);
    glClear(GL_DEPTH_BUFFER_BIT);
}

void gfx::ShadowMap::end() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void gfx::ShadowMap::render_mesh(const Mesh& mesh, const glm::mat4& mvp,
                                 GLuint bone_ssbo)
{
    auto& prog = get_depth_prog();
    prog.use();

    GLint loc = prog.uniform_location("u_mvp");
    if (loc >= 0) prog.uniform_matrix4fv(loc, glm::value_ptr(mvp));

    bool has_skin = bone_ssbo != 0 && mesh.has_bone_data();
    loc = prog.uniform_location("u_has_skin");
    if (loc >= 0) prog.uniform1i(loc, has_skin ? 1 : 0);

    if (has_skin) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, bone_ssbo);
    }

    mesh.draw();
}

void gfx::ShadowMap::bind(int unit) const {
    glBindTextureUnit(unit, handle_);
}

glm::mat4 gfx::compute_light_vp(const glm::mat4& /*camera_vp*/,
                                const glm::vec3& light_dir,
                                float scene_radius,
                                float cascade_distance)
{
    (void)cascade_distance; // distance derived from scene_radius below

    // Place the light well outside the scene bounding sphere
    float light_dist = glm::max(scene_radius * 3.0f, 10.0f);
    glm::vec3 light_pos = -glm::normalize(light_dir) * light_dist;
    glm::mat4 light_view = glm::lookAt(light_pos, glm::vec3(0), glm::vec3(0, 1, 0));

    // Tight XY around the scene, far plane well past it
    float extent = glm::max(scene_radius, 0.1f);
    glm::mat4 light_proj = glm::ortho(-extent, extent, -extent, extent,
                                       0.1f, light_dist * 2.5f);
    return light_proj * light_view;
}
