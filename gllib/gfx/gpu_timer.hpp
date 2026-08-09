#pragma once

#include <cstdint>

namespace gfx {

/// Convenience wrapper around a pair of GL timer queries for measuring GPU time.
/// Automatically handles begin/end/available polling.
class GpuTimer {
public:
    GpuTimer();
    ~GpuTimer();

    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    GpuTimer(GpuTimer&& other) noexcept;
    GpuTimer& operator=(GpuTimer&& other) noexcept;

    void begin();
    void end();

    /// Returns true if the last completed query result is available.
    bool available() const;

    /// Returns elapsed time in nanoseconds (0 if not yet available).
    uint64_t elapsed_ns() const;

    /// Returns elapsed time in milliseconds (0 if not yet available).
    double elapsed_ms() const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

} // namespace gfx
