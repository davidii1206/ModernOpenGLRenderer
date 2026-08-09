#pragma once

#include <glad/glad.h>
#include "shader.hpp"
#include "program.hpp"

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "shader_include.hpp"

namespace gl {

// Tracks a single shader source file
class ShaderFile {
public:
    ShaderFile() = default;
    ShaderFile(std::string_view path, ShaderType type);

    bool valid() const { return !path_.empty(); }

    const std::string& path() const { return path_; }
    ShaderType type() const { return type_; }
    const std::string& source() const { return source_; }

    // Include directories for resolving #include directives
    void add_include_dir(std::string_view dir);

    // Returns true if file was re-read (modified since last read)
    bool poll();

private:
    std::string path_;
    ShaderType type_ = ShaderType::vertex;
    std::string source_;
    std::filesystem::file_time_type last_write_;
    std::vector<std::string> include_dirs_;
};

// A reloadable GL program assembled from multiple ShaderFiles
class HotReloadProgram {
public:
    HotReloadProgram() = default;

    void add_stage(std::string_view path, ShaderType type);

    // Add include directory for #include resolution (applied to all stages)
    void add_include_dir(std::string_view dir);

    // Poll all stages; recompile + relink if any file changed.
    // Returns true if the program was replaced.
    bool poll();

    // Take ownership of the current program (leaves internal pointer null).
    // Call take_program() after a successful poll() to transfer to a Material.
    std::unique_ptr<gl::Program> take_program();

    GLuint handle() const { return program_ ? program_->handle() : 0; }
    bool valid() const { return program_ != nullptr; }

private:
    std::unique_ptr<gl::Program> program_;
    std::vector<ShaderFile> stages_;
    bool had_program_ = false;
    std::vector<std::string> include_dirs_;
};

// Load a SPIR-V binary from a file on disk
Shader shader_from_spirv_file(ShaderType type, const char* path, const char* entry_point = "main");

} // namespace gl
