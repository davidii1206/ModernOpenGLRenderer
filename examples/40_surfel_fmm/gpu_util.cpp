#include "gpu_util.hpp"

#include <gllib/log.hpp>
#include <GLFW/glfw3.h>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>

namespace sgi {

// --- PassTimer --------------------------------------------------------------

void PassTimer::begin() {
    t0_ = std::chrono::steady_clock::now();
    if (gpu_) q_.begin();
}

void PassTimer::end() {
    if (gpu_) q_.end();
    cpu_ms_ = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0_).count();
    ran_ = true;
    if (gpu_) std::swap(q_, q_prev_);
}

double PassTimer::readback() {
    gpu_ms_ = 0.0;
    if (ran_) {
        if (gpu_) gpu_ms_ = double(q_prev_.result()) * 1.0e-6;   // ns -> ms
        cpu_acc_ += cpu_ms_;
        gpu_acc_ += gpu_ms_;
        n_++;
    }
    win_cpu_acc_ += cpu_ms_;
    win_gpu_acc_ += gpu_ms_;
    win_n_++;
    return gpu_ms_;
}

void PassTimer::flush_window() {
    disp_cpu_ = win_n_ ? win_cpu_acc_ / double(win_n_) : 0.0;
    disp_gpu_ = win_n_ ? win_gpu_acc_ / double(win_n_) : 0.0;
    win_cpu_acc_ = win_gpu_acc_ = 0.0;
    win_n_ = 0;
    cpu_hist_.push_back(float(disp_cpu_));
    if (cpu_hist_.size() > kHist) cpu_hist_.erase(cpu_hist_.begin());
    if (gpu_) {
        gpu_hist_.push_back(float(disp_gpu_));
        if (gpu_hist_.size() > kHist) gpu_hist_.erase(gpu_hist_.begin());
    }
}

// --- ImGui diagnostics ------------------------------------------------------

void imgui_stacked_bar(const ImVec2& pos, const ImVec2& size,
                       const float* vals, const ImU32* cols, int n) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    float total = 0.0f;
    for (int i = 0; i < n; ++i) total += vals[i];
    if (total <= 0.0f) {
        dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y),
                          IM_COL32(40, 40, 40, 255));
        return;
    }
    float x = pos.x;
    for (int i = 0; i < n; ++i) {
        float w = size.x * vals[i] / total;
        if (w > 0.0f)
            dl->AddRectFilled(ImVec2(x, pos.y), ImVec2(x + w, pos.y + size.y), cols[i]);
        x += w;
    }
    dl->AddRect(pos, ImVec2(pos.x + size.x, pos.y + size.y), IM_COL32(255, 255, 255, 90));
}

void imgui_stacked_legend(const char* id, const char* const* names,
                          const float* ms, const ImU32* cols, int n, float total) {
    if (ImGui::BeginTable(id, 2)) {
        for (int i = 0; i < n; ++i) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImVec4 c = ImGui::ColorConvertU32ToFloat4(cols[i]);
            ImGui::ColorButton("##sw", c,
                               ImGuiColorEditFlags_NoTooltip |
                               ImGuiColorEditFlags_NoInputs |
                               ImGuiColorEditFlags_NoPicker, ImVec2(10, 10));
            ImGui::SameLine();
            ImGui::Text("%s", names[i]);
            ImGui::TableNextColumn();
            ImGui::Text("%.3f ms  (%.1f%%)", ms[i],
                        total > 0.0f ? 100.0f * ms[i] / total : 0.0f);
        }
        ImGui::EndTable();
    }
}

// --- Pipeline ---------------------------------------------------------------

namespace {
constexpr const char* kIncludeDir = "shaders/common";
}

Pipeline Pipeline::compute(const char* path) {
    Pipeline p;
    p.hot_.add_include_dir(kIncludeDir);
    p.hot_.add_stage(path, gl::ShaderType::compute);
    if (!p.poll())
        gllib::logf(gllib::LogLevel::error, "failed to build compute shader '%s'", path);
    return p;
}

Pipeline Pipeline::raster(const char* vert, const char* frag) {
    Pipeline p;
    p.hot_.add_include_dir(kIncludeDir);
    p.hot_.add_stage(vert, gl::ShaderType::vertex);
    p.hot_.add_stage(frag, gl::ShaderType::fragment);
    if (!p.poll())
        gllib::logf(gllib::LogLevel::error, "failed to build program '%s' + '%s'", vert, frag);
    return p;
}

bool Pipeline::poll() {
    if (!hot_.poll()) return false;
    // Only replace on success: a failed recompile keeps the last good program
    // so a typo while hot-reloading does not black out the frame.
    auto next = hot_.take_program();
    if (!next) return false;
    prog_ = std::move(next);
    return true;
}

#define SGI_UNIFORM(expr)                              \
    do {                                                  \
        if (!prog_) return;                               \
        GLint loc = prog_->uniform_location(name);        \
        if (loc < 0) return;                              \
        expr;                                             \
    } while (0)

void Pipeline::set(const char* name, int v) const        { SGI_UNIFORM(prog_->uniform1i(loc, v)); }
void Pipeline::set(const char* name, unsigned v) const   { SGI_UNIFORM(prog_->uniform1ui(loc, v)); }
void Pipeline::set(const char* name, float v) const      { SGI_UNIFORM(prog_->uniform1f(loc, v)); }
void Pipeline::set(const char* name, const glm::vec2& v) const  { SGI_UNIFORM(prog_->uniform2f(loc, v.x, v.y)); }
void Pipeline::set(const char* name, const glm::vec3& v) const  { SGI_UNIFORM(prog_->uniform3f(loc, v.x, v.y, v.z)); }
void Pipeline::set(const char* name, const glm::vec4& v) const  { SGI_UNIFORM(prog_->uniform4f(loc, v.x, v.y, v.z, v.w)); }
void Pipeline::set(const char* name, const glm::ivec2& v) const { SGI_UNIFORM(prog_->uniform2iv(loc, glm::value_ptr(v))); }
void Pipeline::set(const char* name, const glm::ivec3& v) const { SGI_UNIFORM(prog_->uniform3iv(loc, glm::value_ptr(v))); }
void Pipeline::set(const char* name, const glm::mat3& v) const  { SGI_UNIFORM(prog_->uniform_matrix3fv(loc, glm::value_ptr(v))); }
void Pipeline::set(const char* name, const glm::mat4& v) const  { SGI_UNIFORM(prog_->uniform_matrix4fv(loc, glm::value_ptr(v))); }

#undef SGI_UNIFORM

// --- Misc -------------------------------------------------------------------

void camera_control(gfx::Window& w, gfx::Camera& cam, float dt, bool allow, bool& captured) {
    static double px0 = 0, py0 = 0;
    static bool first = true;

    const bool rmb = w.mouse_down(gfx::MouseButton::right);
    if (allow && rmb) {
        if (!captured) {
            glfwSetInputMode((GLFWwindow*)w.native_handle(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            captured = true;
            first = true;
        }
    } else if (captured) {
        glfwSetInputMode((GLFWwindow*)w.native_handle(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        captured = false;
    }

    if (captured) {
        double cx, cy;
        w.cursor_position(cx, cy);
        if (first) { px0 = cx; py0 = cy; first = false; }
        cam.orbit(float(cx - px0) * 0.008f, float(cy - py0) * 0.008f);
        px0 = cx;
        py0 = cy;
    }

    if (allow) {
        const double scroll = w.scroll_delta();
        if (scroll != 0.0) cam.zoom(float(-scroll) * 0.05f);
    }

    // Movement speed scales with orbit distance so the same controls work on a
    // 2-unit Cornell box and on Sponza.
    const float dist = glm::length(cam.target() - cam.position());
    const float speed = std::max(0.2f, dist) * 0.9f * dt;
    const glm::vec3 fwd = glm::normalize(cam.target() - cam.position());
    const glm::vec3 right = glm::normalize(glm::cross(fwd, glm::vec3(0, 1, 0)));
    glm::vec3 vel(0.0f);
    if (w.key_down(gfx::Key::w)) vel += fwd;
    if (w.key_down(gfx::Key::s)) vel -= fwd;
    if (w.key_down(gfx::Key::a)) vel -= right;
    if (w.key_down(gfx::Key::d)) vel += right;
    if (w.key_down(gfx::Key::space)) vel.y += 1;
    if (w.key_down(gfx::Key::shift)) vel.y -= 1;
    if (glm::length(vel) > 0.0f) cam.set_target(cam.target() + glm::normalize(vel) * speed);

    cam.orbit(0.0f, 0.0f);   // re-derive the view matrix
}

void clear_uint_buffer(gl::Buffer& b) {
    const uint32_t zero = 0;
    b.clear(GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
}

} // namespace sgi
