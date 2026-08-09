#pragma once

#include <glad/glad.h>

namespace gl {

enum class SyncStatus {
    signaled    = GL_SIGNALED,
    unsignaled  = GL_UNSIGNALED,
    already_signaled = GL_ALREADY_SIGNALED,
    timeout_expired  = GL_TIMEOUT_EXPIRED,
    condition_satisfied = GL_CONDITION_SATISFIED,
    wait_failed  = GL_WAIT_FAILED,
};

class Sync {
public:
    Sync();
    ~Sync();

    Sync(const Sync&) = delete;
    Sync& operator=(const Sync&) = delete;

    Sync(Sync&& other) noexcept;
    Sync& operator=(Sync&& other) noexcept;

    SyncStatus client_wait(GLuint64 timeout_ns) const;
    void wait_on_gpu() const;

    GLsync handle() const { return handle_; }

private:
    GLsync handle_ = nullptr;
};

} // namespace gl
