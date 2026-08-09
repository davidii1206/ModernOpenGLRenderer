#include "atomic_counter.hpp"

#include <cstring>

namespace gl {

AtomicCounter::AtomicCounter() {
    glCreateBuffers(1, &handle_);
}

AtomicCounter::~AtomicCounter() {
    if (handle_) glDeleteBuffers(1, &handle_);
}

AtomicCounter::AtomicCounter(AtomicCounter&& other) noexcept
    : handle_(other.handle_), size_(other.size_)
{
    other.handle_ = 0;
    other.size_ = 0;
}

AtomicCounter& AtomicCounter::operator=(AtomicCounter&& other) noexcept {
    if (this != &other) {
        if (handle_) glDeleteBuffers(1, &handle_);
        handle_ = other.handle_;
        size_ = other.size_;
        other.handle_ = 0;
        other.size_ = 0;
    }
    return *this;
}

void AtomicCounter::data(GLsizeiptr size, const void* initial) {
    size_ = size;
    glNamedBufferData(handle_, size, initial, GL_DYNAMIC_COPY);
}

void AtomicCounter::sub_data(const void* data, GLintptr offset, GLsizeiptr size) {
    glNamedBufferSubData(handle_, offset, size, data);
}

void AtomicCounter::bind(GLuint index) const {
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, index, handle_);
}

void AtomicCounter::reset(GLuint value) {
    glClearNamedBufferData(handle_, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &value);
}

} // namespace gl
