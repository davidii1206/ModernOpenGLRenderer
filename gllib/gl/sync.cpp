#include "sync.hpp"

#include <gllib/log.hpp>

namespace gl {

Sync::Sync() {
    handle_ = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}

Sync::~Sync() {
    if (handle_) {
        glDeleteSync(handle_);
    }
}

Sync::Sync(Sync&& other) noexcept
    : handle_(other.handle_)
{
    other.handle_ = nullptr;
}

Sync& Sync::operator=(Sync&& other) noexcept {
    if (this != &other) {
        if (handle_) {
            glDeleteSync(handle_);
        }
        handle_ = other.handle_;
        other.handle_ = nullptr;
    }
    return *this;
}

SyncStatus Sync::client_wait(GLuint64 timeout_ns) const {
    if (!handle_) return SyncStatus::wait_failed;
    GLenum result = glClientWaitSync(handle_, 0, timeout_ns);
    if (result == GL_WAIT_FAILED) {
        gllib::log(gllib::LogLevel::error, "glClientWaitSync failed");
    }
    return static_cast<SyncStatus>(result);
}

void Sync::wait_on_gpu() const {
    if (handle_) {
        glWaitSync(handle_, 0, GL_TIMEOUT_IGNORED);
    }
}

} // namespace gl
