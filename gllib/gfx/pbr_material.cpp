#include "pbr_material.hpp"
#include "ibl_probe.hpp"
#include "shadow_map.hpp"
#include "cubemap.hpp"
#include "model.hpp"
#include "mesh.hpp"
#include "texture.hpp"
#include "../gl/shader.hpp"
#include "../gl/program.hpp"
#include "../gl/texture.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cmath>

static const char* pbr_vs = R"(
#version 430 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 4) in uvec4 a_bone_indices;
layout(location = 5) in vec4 a_bone_weights;

layout(std430, binding = 4) readonly buffer BoneMatrices {
    mat4 u_bone_matrices[];
};

uniform mat4 u_model;
uniform mat4 u_view_proj;
uniform mat3 u_normal_mat;
uniform mat4 u_light_vp;
uniform bool u_has_skin;

out vec3 v_pos;
out vec3 v_normal;
out vec2 v_uv;
out vec4 v_frag_light_space;

void main() {
    vec4 pos = vec4(a_pos, 1.0);
    vec3 nrm = a_normal;

    if (u_has_skin) {
        mat4 skin = mat4(0);
        for (int i = 0; i < 4; ++i) {
            skin += a_bone_weights[i] * u_bone_matrices[a_bone_indices[i]];
        }
        pos = skin * pos;
        nrm = mat3(skin) * nrm;
    }

    vec4 world_pos = u_model * pos;
    gl_Position = u_view_proj * world_pos;
    v_frag_light_space = u_light_vp * world_pos;
    v_pos = world_pos.xyz;
    v_normal = normalize(u_normal_mat * nrm);
    v_uv = a_uv;
}
)";

static const char* pbr_fs = R"(
#version 430 core
in vec3 v_pos;
in vec3 v_normal;
in vec2 v_uv;
in vec4 v_frag_light_space;

layout(location = 0) out vec4 frag_color;

uniform vec3 u_view_pos;
uniform vec3 u_albedo;
uniform float u_metallic;
uniform float u_roughness;
uniform float u_ao;
uniform int u_alpha_mode;
uniform float u_alpha_cutoff;

uniform bool u_has_base_tex;
uniform bool u_has_mr_tex;
uniform bool u_has_occlusion_tex;

uniform sampler2D u_base_tex;
uniform sampler2D u_mr_tex;
uniform sampler2D u_occlusion_tex;

uniform samplerCube u_irradiance_map;
uniform samplerCube u_prefilter_map;
uniform sampler2D u_brdf_lut;
uniform int u_prefilter_levels;

uniform sampler2D u_shadow_map;
uniform bool u_has_shadow;
uniform vec2 u_shadow_map_size;

uniform vec3 u_ambient_top;
uniform vec3 u_ambient_bottom;
uniform float u_ambient_intensity;

vec3 aces(vec3 x) {
    float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 fresnel_schlick_roughness(float cos_theta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(max(1.0 - cos_theta, 0.0), 5.0);
}

float shadow_factor(vec4 frag_light_space, sampler2D shadow_map) {
    vec3 proj = frag_light_space.xyz / frag_light_space.w;
    proj = proj * 0.5 + 0.5;
    if (proj.x < 0 || proj.x > 1 || proj.y < 0 || proj.y > 1 || proj.z < 0 || proj.z > 1)
        return 1.0;
    float bias = 0.001;
    return (proj.z - bias) > texture(shadow_map, proj.xy).r ? 0.0 : 1.0;
}

void main() {
    vec4 base = u_has_base_tex ? texture(u_base_tex, v_uv) : vec4(1.0);
    base.rgb *= u_albedo;

    if (u_alpha_mode == 1 && base.a < u_alpha_cutoff) discard;

    vec3 N = normalize(v_normal);

    vec3 V = normalize(u_view_pos - v_pos);
    float NdotV = max(dot(N, V), 0.0);

    vec2 mr = u_has_mr_tex ? texture(u_mr_tex, v_uv).bg : vec2(1.0);
    float metal = mr.x * u_metallic;
    float rough = max(mr.y * u_roughness, 0.001);
    float ao = u_has_occlusion_tex ? texture(u_occlusion_tex, v_uv).r * u_ao : u_ao;

    vec3 F0 = mix(vec3(0.04), base.rgb, metal);
    vec3 kS = fresnel_schlick_roughness(NdotV, F0, rough);
    vec3 kD = (1.0 - kS) * (1.0 - metal);

    vec3 irradiance = texture(u_irradiance_map, N).rgb;
    vec3 diffuse = kD * irradiance * base.rgb;

    vec3 R = reflect(-V, N);
    float level = rough * float(u_prefilter_levels - 1);
    vec3 prefilter = textureLod(u_prefilter_map, R, level).rgb;
    vec2 brdf = texture(u_brdf_lut, vec2(NdotV, rough)).rg;
    vec3 specular = prefilter * (kS * brdf.x + brdf.y);

    vec3 ambient = (diffuse + specular) * ao;

    vec3 hemi = mix(u_ambient_bottom, u_ambient_top, N.y * 0.5 + 0.5);
    ambient += hemi * u_ambient_intensity;

    float shadow = 1.0;
    if (u_has_shadow)
        shadow = shadow_factor(v_frag_light_space, u_shadow_map);
    ambient *= mix(0.15, 1.0, shadow);

    vec3 color = aces(ambient);
    color = pow(color, vec3(1.0 / 2.2));

    frag_color = vec4(color, u_alpha_mode == 2 ? base.a : 1.0);
}
)";

gfx::PBRMaterial::PBRMaterial() {
    prog_ = new gl::Program;
    gl::Shader vs(gl::ShaderType::vertex, pbr_vs);
    gl::Shader fs(gl::ShaderType::fragment, pbr_fs);
    if (vs.compiled() && fs.compiled()) {
        prog_->attach(vs);
        prog_->attach(fs);
        valid_ = prog_->link();
    }
    if (!valid_) {
        delete prog_;
        prog_ = nullptr;
    }
}

gfx::PBRMaterial::~PBRMaterial() { delete prog_; }

gfx::PBRMaterial::PBRMaterial(PBRMaterial&& other) noexcept
    : prog_(other.prog_), valid_(other.valid_)
{
    other.prog_ = nullptr; other.valid_ = false;
}

gfx::PBRMaterial& gfx::PBRMaterial::operator=(PBRMaterial&& other) noexcept {
    if (this != &other) {
        delete prog_;
        prog_ = other.prog_; valid_ = other.valid_;
        other.prog_ = nullptr; other.valid_ = false;
    }
    return *this;
}

void gfx::PBRMaterial::begin(const glm::mat4& view, const glm::mat4& proj, const glm::vec3& view_pos) {
    if (!valid_) return;
    prog_->use();
    auto uloc = [&](const char* n) { return prog_->uniform_location(n); };
    glm::mat4 vp = proj * view;
    GLint loc;
    loc = uloc("u_view_proj");   if (loc >= 0) prog_->uniform_matrix4fv(loc, glm::value_ptr(vp));
    loc = uloc("u_view_pos");    if (loc >= 0) prog_->uniform3fv(loc, glm::value_ptr(view_pos));
}

void gfx::PBRMaterial::set_model_matrix(const glm::mat4& model) {
    if (!valid_) return;
    auto uloc = [&](const char* n) { return prog_->uniform_location(n); };
    GLint loc = uloc("u_model");
    if (loc >= 0) prog_->uniform_matrix4fv(loc, glm::value_ptr(model));
    glm::mat3 nm = glm::transpose(glm::inverse(glm::mat3(model)));
    loc = uloc("u_normal_mat");
    if (loc >= 0) prog_->uniform_matrix3fv(loc, glm::value_ptr(nm));
}

void gfx::PBRMaterial::set_ibl(const IBLProbe& ibl) {
    if (!valid_) return;
    auto uloc = [&](const char* n) { return prog_->uniform_location(n); };
    ibl.irradiance_map().bind(4);
    ibl.prefilter_map().bind(5);
    ibl.brdf_lut().bind(6);
    GLint loc;
    loc = uloc("u_irradiance_map");  if (loc >= 0) prog_->uniform1i(loc, 4);
    loc = uloc("u_prefilter_map");   if (loc >= 0) prog_->uniform1i(loc, 5);
    loc = uloc("u_brdf_lut");        if (loc >= 0) prog_->uniform1i(loc, 6);
    loc = uloc("u_prefilter_levels"); if (loc >= 0) prog_->uniform1i(loc, ibl.prefilter_map().levels());
}

void gfx::PBRMaterial::set_shadow(const ShadowMap& shadow, const glm::mat4& light_vp) {
    if (!valid_) return;
    auto uloc = [&](const char* n) { return prog_->uniform_location(n); };
    shadow.bind(7);
    GLint loc;
    loc = uloc("u_shadow_map");      if (loc >= 0) prog_->uniform1i(loc, 7);
    loc = uloc("u_light_vp");        if (loc >= 0) prog_->uniform_matrix4fv(loc, glm::value_ptr(light_vp));
    loc = uloc("u_has_shadow");      if (loc >= 0) prog_->uniform1i(loc, 1);
    float sm_size[2] = {float(shadow.size()), float(shadow.size())};
    loc = uloc("u_shadow_map_size"); if (loc >= 0) prog_->uniform2fv(loc, sm_size);
}

void gfx::PBRMaterial::set_material(const ModelMaterialInfo& mat, const Model& model) {
    if (!valid_) return;
    auto uloc = [&](const char* n) { return prog_->uniform_location(n); };
    GLint loc;

    loc = uloc("u_albedo"); if (loc >= 0) prog_->uniform3fv(loc, mat.base_color_factor);

    if (mat.base_color_tex >= 0 && size_t(mat.base_color_tex) < model.texture_count()) {
        model.texture(mat.base_color_tex)->bind(0);
        loc = uloc("u_base_tex");     if (loc >= 0) prog_->uniform1i(loc, 0);
        loc = uloc("u_has_base_tex"); if (loc >= 0) prog_->uniform1i(loc, 1);
    } else {
        loc = uloc("u_has_base_tex"); if (loc >= 0) prog_->uniform1i(loc, 0);
    }

    if (mat.metallic_roughness_tex >= 0 && size_t(mat.metallic_roughness_tex) < model.texture_count()) {
        model.texture(mat.metallic_roughness_tex)->bind(1);
        loc = uloc("u_mr_tex");     if (loc >= 0) prog_->uniform1i(loc, 1);
        loc = uloc("u_has_mr_tex"); if (loc >= 0) prog_->uniform1i(loc, 1);
    } else {
        loc = uloc("u_has_mr_tex"); if (loc >= 0) prog_->uniform1i(loc, 0);
    }
    loc = uloc("u_metallic"); if (loc >= 0) prog_->uniform1f(loc, mat.metallic_factor);
    loc = uloc("u_roughness"); if (loc >= 0) prog_->uniform1f(loc, mat.roughness_factor);

    if (mat.normal_tex >= 0 && size_t(mat.normal_tex) < model.texture_count()) {
        model.texture(mat.normal_tex)->bind(2);
        loc = uloc("u_normal_tex");     if (loc >= 0) prog_->uniform1i(loc, 2);
        loc = uloc("u_has_normal_tex"); if (loc >= 0) prog_->uniform1i(loc, 1);
    } else {
        loc = uloc("u_has_normal_tex"); if (loc >= 0) prog_->uniform1i(loc, 0);
    }

    if (mat.occlusion_tex >= 0 && size_t(mat.occlusion_tex) < model.texture_count()) {
        model.texture(mat.occlusion_tex)->bind(3);
        loc = uloc("u_occlusion_tex");     if (loc >= 0) prog_->uniform1i(loc, 3);
        loc = uloc("u_has_occlusion_tex"); if (loc >= 0) prog_->uniform1i(loc, 1);
    } else {
        loc = uloc("u_has_occlusion_tex"); if (loc >= 0) prog_->uniform1i(loc, 0);
    }

    loc = uloc("u_ao");           if (loc >= 0) prog_->uniform1f(loc, 1.0f);
    loc = uloc("u_alpha_mode");   if (loc >= 0) prog_->uniform1i(loc, int(mat.alpha_mode));
    loc = uloc("u_alpha_cutoff"); if (loc >= 0) prog_->uniform1f(loc, mat.alpha_cutoff);
}

void gfx::PBRMaterial::set_skin(GLuint bone_ssbo) {
    if (!valid_) return;
    auto uloc = [&](const char* n) { return prog_->uniform_location(n); };
    if (bone_ssbo != 0) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, bone_ssbo);
        GLint loc = uloc("u_has_skin");
        if (loc >= 0) prog_->uniform1i(loc, 1);
    } else {
        GLint loc = uloc("u_has_skin");
        if (loc >= 0) prog_->uniform1i(loc, 0);
    }
}

void gfx::PBRMaterial::set_ambient_hemi(const glm::vec3& top, const glm::vec3& bottom, float intensity) {
    if (!valid_) return;
    auto uloc = [&](const char* n) { return prog_->uniform_location(n); };
    GLint loc;
    loc = uloc("u_ambient_top");     if (loc >= 0) prog_->uniform3fv(loc, glm::value_ptr(top));
    loc = uloc("u_ambient_bottom");  if (loc >= 0) prog_->uniform3fv(loc, glm::value_ptr(bottom));
    loc = uloc("u_ambient_intensity"); if (loc >= 0) prog_->uniform1f(loc, intensity);
}

void gfx::PBRMaterial::draw(const Mesh& mesh) {
    if (!valid_) return;
    mesh.draw();
}

void gfx::PBRMaterial::end() {
    glUseProgram(0);
}
