#pragma once

// ---------------------------------------------------------------------------
// Shared GPU plumbing: per-pass timing, ImGui diagnostics widgets, shader
// loading, and the exclusive-prefix-sum driver every grid build needs.
// ---------------------------------------------------------------------------

#include <gl/gl.hpp>
#include <gfx/gfx.hpp>
#include <imgui.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace mosaic {

// --- Timing ----------------------------------------------------------------

// Per-pass GPU/CPU timing with double-buffered GL_TIME_ELAPSED queries.
// Only one time-elapsed query may be active at a time, so passes are timed
// sequentially and never nested; the frame timer is CPU-only.
class PassTimer {
public:
    explicit PassTimer(const char* name, bool gpu = true)
        : name_(name), gpu_(gpu), q_(gl::QueryType::time_elapsed),
          q_prev_(gl::QueryType::time_elapsed) {}

    void begin();
    void end();
    void skip() { ran_ = false; cpu_ms_ = 0.0; }

    // Supplies an externally measured duration instead of timing a scope.
    //
    // Needed for the frame timer: timing the CPU span of the frame loop measures
    // how long it takes to SUBMIT the work, not how long the frame takes. With
    // vsync off and no sync point the CPU runs far ahead of the GPU, so that
    // number can read 1 ms while the GPU is taking 100. Wall-clock delta between
    // successive frames is the honest figure.
    void submit_external(double ms) { cpu_ms_ = ms; ran_ = true; }

    // Reads the PREVIOUS dispatch's query (already signalled, so the read
    // cannot stall on a live query) and accumulates into the display window.
    double readback();

    // Called every ~0.5 s: publishes the windowed average and appends to the
    // history plot, so the readout is stable enough to read while tuning.
    void flush_window();

    const char* name() const { return name_; }
    bool gpu() const { return gpu_; }
    double cpu_ms() const { return cpu_ms_; }
    double gpu_ms() const { return gpu_ms_; }
    double disp_cpu() const { return disp_cpu_; }
    double disp_gpu() const { return disp_gpu_; }
    double avg_cpu() const { return n_ ? cpu_acc_ / double(n_) : 0.0; }
    double avg_gpu() const { return n_ ? gpu_acc_ / double(n_) : 0.0; }
    const std::vector<float>& gpu_hist() const { return gpu_hist_; }

private:
    static constexpr size_t kHist = 120;
    const char* name_;
    bool gpu_;
    gl::Query q_, q_prev_;
    std::chrono::steady_clock::time_point t0_;
    double cpu_ms_ = 0.0, gpu_ms_ = 0.0;
    double cpu_acc_ = 0.0, gpu_acc_ = 0.0;
    double win_cpu_acc_ = 0.0, win_gpu_acc_ = 0.0;
    double disp_cpu_ = 0.0, disp_gpu_ = 0.0;
    int n_ = 0, win_n_ = 0;
    bool ran_ = false;
    std::vector<float> cpu_hist_, gpu_hist_;
};

// RAII scope so a pass cannot be left un-ended on an early return.
struct ScopedPass {
    PassTimer& t;
    explicit ScopedPass(PassTimer& timer) : t(timer) { t.begin(); }
    ~ScopedPass() { t.end(); }
};

// --- ImGui diagnostics ------------------------------------------------------

void imgui_stacked_bar(const ImVec2& pos, const ImVec2& size,
                       const float* vals, const ImU32* cols, int n);

void imgui_stacked_legend(const char* id, const char* const* names,
                          const float* ms, const ImU32* cols, int n, float total);

// --- Shader loading ---------------------------------------------------------

// A hot-reloadable program plus the live gl::Program it produced. All stage
// paths are relative to the working directory (the example's build dir), and
// shaders/common is registered as an include dir so shaders can #include the
// shared GLSL headers instead of duplicating them.
class Pipeline {
public:
    Pipeline() = default;

    // Compute pipeline from a single .comp file.
    static Pipeline compute(const char* path);
    // Raster pipeline from a vertex + fragment file pair.
    static Pipeline raster(const char* vert, const char* frag);

    // Recompile if any source changed. Returns true when the program was
    // replaced (safe to call every frame).
    bool poll();

    gl::Program* operator->() const { return prog_.get(); }
    gl::Program& operator*() const { return *prog_; }
    bool valid() const { return prog_ != nullptr; }
    void use() const { prog_->use(); }

    // Uniform setters that tolerate a missing/optimised-out uniform, so a
    // shader can drop an input without breaking the host code.
    void set(const char* name, int v) const;
    void set(const char* name, unsigned v) const;
    void set(const char* name, float v) const;
    void set(const char* name, const glm::vec2& v) const;
    void set(const char* name, const glm::vec3& v) const;
    void set(const char* name, const glm::vec4& v) const;
    void set(const char* name, const glm::ivec2& v) const;
    void set(const char* name, const glm::ivec3& v) const;
    void set(const char* name, const glm::mat3& v) const;
    void set(const char* name, const glm::mat4& v) const;

private:
    gl::HotReloadProgram hot_;
    std::unique_ptr<gl::Program> prog_;
};

// --- Prefix sum -------------------------------------------------------------

// Exclusive prefix sum over a segment of a uint SSBO, as the count -> start
// transform every counting-sort grid build needs.
//
// Block size 256, and the middle kernel scans all block sums in ONE workgroup,
// which caps a single scan at kScanBlocksMax * kBlockSize elements. That is
// 262144 = 64^3, exactly the clipmap cascade size — there is no headroom, so a
// larger cascade needs a multi-level scan here first.
class PrefixSum {
public:
    static constexpr uint32_t kBlockSize = 256;
    static constexpr uint32_t kScanBlocksMax = 1024;
    static constexpr uint32_t kMaxElements = kBlockSize * kScanBlocksMax;

    bool init();
    bool poll();

    // counts[off .. off+n) -> starts[off .. off+n), exclusive.
    void run(gl::Buffer& counts, gl::Buffer& starts, uint32_t n, uint32_t off = 0);

private:
    Pipeline block_, blocks_, finish_;
    gl::Buffer block_sums_{gl::BufferType::shader, gl::BufferUsage::dynamic_draw};
};

// --- Misc -------------------------------------------------------------------

// Orbit camera: right-drag orbits, scroll zooms, WASD + space/shift pan.
void camera_control(gfx::Window& w, gfx::Camera& cam, float dt, bool allow, bool& captured);

// Zero a uint SSBO (or a prefix of one) via glClearNamedBufferData.
void clear_uint_buffer(gl::Buffer& b);

} // namespace mosaic
