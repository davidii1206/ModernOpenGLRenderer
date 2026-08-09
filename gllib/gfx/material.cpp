#include "material.hpp"
#include "texture.hpp"
#include <gl/program.hpp>

namespace gfx {

Material::~Material() = default;

Material::Material(Material&& other) noexcept
    : program_(std::move(other.program_))
    , textures_(std::move(other.textures_))
    , float_uniforms_(std::move(other.float_uniforms_))
    , int_uniforms_(std::move(other.int_uniforms_))
    , uint_uniforms_(std::move(other.uint_uniforms_))
    , vec2_uniforms_(std::move(other.vec2_uniforms_))
    , vec3_uniforms_(std::move(other.vec3_uniforms_))
    , vec4_uniforms_(std::move(other.vec4_uniforms_))
    , mat4_uniforms_(std::move(other.mat4_uniforms_))
{
}

Material& Material::operator=(Material&& other) noexcept {
    if (this != &other) {
        program_ = std::move(other.program_);
        textures_ = std::move(other.textures_);
        float_uniforms_ = std::move(other.float_uniforms_);
        int_uniforms_ = std::move(other.int_uniforms_);
        uint_uniforms_ = std::move(other.uint_uniforms_);
        vec2_uniforms_ = std::move(other.vec2_uniforms_);
        vec3_uniforms_ = std::move(other.vec3_uniforms_);
        vec4_uniforms_ = std::move(other.vec4_uniforms_);
        mat4_uniforms_ = std::move(other.mat4_uniforms_);
    }
    return *this;
}

void Material::set_program(std::unique_ptr<gl::Program> program) {
    program_ = std::move(program);
}

void Material::set_texture(const std::string& name, std::shared_ptr<Texture> texture) {
    textures_[name] = std::move(texture);
}

void Material::set_uniform(const std::string& name, float value) {
    float_uniforms_[name] = value;
}

void Material::set_uniform(const std::string& name, int value) {
    int_uniforms_[name] = value;
}

void Material::set_uniform(const std::string& name, unsigned int value) {
    uint_uniforms_[name] = value;
}

void Material::set_uniform(const std::string& name, const glm::vec2& value) {
    vec2_uniforms_[name] = value;
}

void Material::set_uniform(const std::string& name, const glm::vec3& value) {
    vec3_uniforms_[name] = value;
}

void Material::set_uniform(const std::string& name, const glm::vec4& value) {
    vec4_uniforms_[name] = value;
}

void Material::set_uniform(const std::string& name, const glm::mat4& value) {
    mat4_uniforms_[name] = value;
}

void Material::bind() const {
    if (!program_) return;

    program_->use();

    for (const auto& [name, value] : float_uniforms_) {
        auto loc = program_->uniform_location(name);
        if (loc >= 0) program_->uniform1f(loc, value);
    }

    for (const auto& [name, value] : int_uniforms_) {
        auto loc = program_->uniform_location(name);
        if (loc >= 0) program_->uniform1i(loc, value);
    }

    for (const auto& [name, value] : uint_uniforms_) {
        auto loc = program_->uniform_location(name);
        if (loc >= 0) program_->uniform1ui(loc, value);
    }

    for (const auto& [name, value] : vec2_uniforms_) {
        auto loc = program_->uniform_location(name);
        if (loc >= 0) program_->uniform2fv(loc, &value[0]);
    }

    for (const auto& [name, value] : vec3_uniforms_) {
        auto loc = program_->uniform_location(name);
        if (loc >= 0) program_->uniform3fv(loc, &value[0]);
    }

    for (const auto& [name, value] : vec4_uniforms_) {
        auto loc = program_->uniform_location(name);
        if (loc >= 0) program_->uniform4fv(loc, &value[0]);
    }

    for (const auto& [name, value] : mat4_uniforms_) {
        auto loc = program_->uniform_location(name);
        if (loc >= 0) program_->uniform_matrix4fv(loc, &value[0][0]);
    }

    int unit = 0;
    for (const auto& [name, texture] : textures_) {
        texture->bind(unit);
        auto loc = program_->uniform_location(name);
        if (loc >= 0) program_->uniform1i(loc, unit);
        ++unit;
    }
}

void Material::unbind() const {
    gl::Program::unuse();
}

GLuint Material::program_handle() const {
    return program_ ? program_->handle() : 0;
}

} // namespace gfx
