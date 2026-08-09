#include "shader_file.hpp"

#include <gllib/log.hpp>

#include <string>
#include <fstream>
#include <vector>

namespace gl {
namespace {

std::string read_file(const char* path) {
    std::ifstream file(path, std::ios::in | std::ios::binary | std::ios::ate);
    if (!file) return {};
    auto size = file.tellg();
    std::string content(static_cast<std::size_t>(size), '\0');
    file.seekg(0);
    file.read(content.data(), size);
    return content;
}

} // namespace

Shader shader_from_file(ShaderType type, const char* path) {
    auto source = read_file(path);
    if (source.empty()) {
        gllib::logf(gllib::LogLevel::error, "failed to read shader file: %s", path);
        return {};
    }
    return Shader(type, source);
}

Program program_from_files(const char* vert_path, const char* frag_path) {
    auto vert = shader_from_file(ShaderType::vertex, vert_path);
    if (!vert.valid() || !vert.compiled()) return {};

    auto frag = shader_from_file(ShaderType::fragment, frag_path);
    if (!frag.valid() || !frag.compiled()) return {};

    Program prog;
    prog.attach(vert);
    prog.attach(frag);
    prog.link();
    return prog;
}

Program program_from_files(const char* vert_path, const char* frag_path,
                            const char* geom_path)
{
    auto vert = shader_from_file(ShaderType::vertex, vert_path);
    if (!vert.valid() || !vert.compiled()) return {};

    auto frag = shader_from_file(ShaderType::fragment, frag_path);
    if (!frag.valid() || !frag.compiled()) return {};

    auto geom = shader_from_file(ShaderType::geometry, geom_path);
    if (!geom.valid() || !geom.compiled()) return {};

    Program prog;
    prog.attach(vert);
    prog.attach(frag);
    prog.attach(geom);
    prog.link();
    return prog;
}

} // namespace gl
