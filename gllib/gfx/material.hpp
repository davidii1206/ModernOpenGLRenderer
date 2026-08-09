#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <unordered_map>

namespace gl {
class Program;
}

namespace gfx {

class Texture;

class Material {
public:
    Material() = default;
    ~Material();

    Material(const Material&) = delete;
    Material& operator=(const Material&) = delete;

    Material(Material&& other) noexcept;
    Material& operator=(Material&& other) noexcept;

    void set_program(std::unique_ptr<gl::Program> program);
    void set_texture(const std::string& name, std::shared_ptr<Texture> texture);
    void set_uniform(const std::string& name, float value);
    void set_uniform(const std::string& name, int value);
    void set_uniform(const std::string& name, unsigned int value);
    void set_uniform(const std::string& name, const glm::vec2& value);
    void set_uniform(const std::string& name, const glm::vec3& value);
    void set_uniform(const std::string& name, const glm::vec4& value);
    void set_uniform(const std::string& name, const glm::mat4& value);

    void bind() const;
    void unbind() const;
    GLuint program_handle() const;

private:
    std::unique_ptr<gl::Program> program_;
    std::unordered_map<std::string, std::shared_ptr<Texture>> textures_;
    std::unordered_map<std::string, float> float_uniforms_;
    std::unordered_map<std::string, int> int_uniforms_;
    std::unordered_map<std::string, unsigned int> uint_uniforms_;
    std::unordered_map<std::string, glm::vec2> vec2_uniforms_;
    std::unordered_map<std::string, glm::vec3> vec3_uniforms_;
    std::unordered_map<std::string, glm::vec4> vec4_uniforms_;
    std::unordered_map<std::string, glm::mat4> mat4_uniforms_;
};

} // namespace gfx
