#include "hot_reload.hpp"

#include <gllib/log.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

namespace gl {

static std::string read_file(const char* path) {
    std::ifstream file(path, std::ios::in | std::ios::binary | std::ios::ate);
    if (!file) return {};
    auto size = file.tellg();
    std::string content(static_cast<std::size_t>(size), '\0');
    file.seekg(0);
    file.read(content.data(), size);
    return content;
}

// --- ShaderFile ---

ShaderFile::ShaderFile(std::string_view path, ShaderType type)
    : path_(path), type_(type)
{
    poll(); // initial read
}

void ShaderFile::add_include_dir(std::string_view dir) {
    include_dirs_.emplace_back(dir);
}

bool ShaderFile::poll() {
    std::error_code ec;
    auto mtime = std::filesystem::last_write_time(path_, ec);
    if (ec) return false;
    // Re-read if mtime changed OR if source_ is empty (first read after setting include dirs)
    if (mtime == last_write_ && !source_.empty()) return false;

    auto res = resolve_includes(path_, include_dirs_);
    if (!res.success) {
        gllib::logf(gllib::LogLevel::error, "ShaderFile: %s: %s", path_.c_str(), res.error.c_str());
        return false;
    }

    source_ = std::move(res.source);
    last_write_ = mtime;
    return true;
}

// --- HotReloadProgram ---

void HotReloadProgram::add_stage(std::string_view path, ShaderType type) {
    stages_.emplace_back(path, type);
}

void HotReloadProgram::add_include_dir(std::string_view dir) {
    include_dirs_.emplace_back(dir);
    for (auto& s : stages_)
        s.add_include_dir(dir);
}

bool HotReloadProgram::poll() {
    // Check if any stage file changed
    bool any_changed = false;
    for (auto& stage : stages_) {
        if (stage.poll()) any_changed = true;
    }
    // Only recompile if sources changed OR we've never successfully compiled
    if (!any_changed && had_program_) return false;

    // Compile all stages from current source
    std::vector<Shader> shaders;
    for (const auto& stage : stages_) {
        // An empty source means the file could not be read (missing path,
        // wrong working directory, failed #include). Bail out with a clear
        // message instead of feeding "" to the driver, which "compiles" it
        // and then fails linking with a cryptic "must write to gl_Position"
        // / "no work group size specified".
        if (stage.source().empty()) {
            gllib::logf(gllib::LogLevel::error,
                "HotReloadProgram: no source for %s (missing file or wrong "
                "working directory?)", stage.path().c_str());
            return false;
        }
        Shader s(stage.type(), stage.source());
        if (!s.compiled()) {
            gllib::logf(gllib::LogLevel::error,
                "HotReloadProgram: failed to compile %s", stage.path().c_str());
            return false;
        }
        shaders.push_back(std::move(s));
    }

    // Link new program
    auto new_prog = std::make_unique<Program>();
    for (const auto& s : shaders) {
        new_prog->attach(s);
    }
    if (!new_prog->link()) {
        gllib::logf(gllib::LogLevel::error,
            "HotReloadProgram: link failed for %zu stages", stages_.size());
        return false;
    }

    gllib::logf(gllib::LogLevel::info,
        "HotReloadProgram: recompiled %zu stages", stages_.size());

    program_ = std::move(new_prog);
    had_program_ = true;
    return true;
}

std::unique_ptr<Program> HotReloadProgram::take_program() {
    return std::move(program_);
}

// --- SPIR-V from file ---

Shader shader_from_spirv_file(ShaderType type, const char* path, const char* entry_point) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        gllib::logf(gllib::LogLevel::error, "spirv_from_file: failed to open %s", path);
        return {};
    }
    auto size = file.tellg();
    std::vector<char> buf(static_cast<std::size_t>(size));
    file.seekg(0);
    file.read(buf.data(), size);

    return Shader::from_spirv(type, buf.data(), buf.size(), entry_point);
}

} // namespace gl
