#include "gpu_timer.hpp"

#include <gl/query.hpp>

#include <cstdio>

namespace gfx {

struct GpuTimer::Impl {
    gl::Query q0{gl::QueryType::time_elapsed};
    gl::Query q1{gl::QueryType::time_elapsed};
    gl::Query* queries[2] = {&q0, &q1};
    int current = 0;
    bool pending = false;
};

GpuTimer::GpuTimer()
    : impl_(new Impl())
{}

GpuTimer::~GpuTimer() {
    delete impl_;
}

GpuTimer::GpuTimer(GpuTimer&& other) noexcept
    : impl_(other.impl_)
{
    other.impl_ = nullptr;
}

GpuTimer& GpuTimer::operator=(GpuTimer&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

void GpuTimer::begin() {
    if (!impl_ || impl_->pending) return;
    impl_->queries[impl_->current]->begin();
}

void GpuTimer::end() {
    if (!impl_ || impl_->pending) return;
    impl_->queries[impl_->current]->end();
    impl_->pending = true;
}

bool GpuTimer::available() const {
    if (!impl_ || !impl_->pending) return false;
    return impl_->queries[impl_->current]->result_available();
}

uint64_t GpuTimer::elapsed_ns() const {
    if (!impl_ || !impl_->pending) return 0;
    uint64_t result = impl_->queries[impl_->current]->result();
    if (result > 0) {
        impl_->pending = false;
        impl_->current = (impl_->current + 1) % 2;
    }
    return result;
}

double GpuTimer::elapsed_ms() const {
    return static_cast<double>(elapsed_ns()) / 1e6;
}

} // namespace gfx
