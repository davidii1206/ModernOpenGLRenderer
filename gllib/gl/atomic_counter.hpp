#pragma once

#include <glad/glad.h>

namespace gl {

class AtomicCounter {
public:
    AtomicCounter();
    ~AtomicCounter();

    AtomicCounter(const AtomicCounter&) = delete;
    AtomicCounter& operator=(const AtomicCounter&) = delete;

    AtomicCounter(AtomicCounter&& other) noexcept;
    AtomicCounter& operator=(AtomicCounter&& other) noexcept;

    void data(GLsizeiptr size, const void* initial = nullptr);
    void sub_data(const void* data, GLintptr offset, GLsizeiptr size);

    void bind(GLuint index) const;

    void reset(GLuint value = 0);

    GLuint handle() const { return handle_; }
    GLsizeiptr size() const { return size_; }

private:
    GLuint handle_ = 0;
    GLsizeiptr size_ = 0;
};

} // namespace gl
