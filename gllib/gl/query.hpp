#pragma once

#include <glad/glad.h>

namespace gl {

enum class QueryType : GLenum {
    time_elapsed       = GL_TIME_ELAPSED,
    samples_passed     = GL_SAMPLES_PASSED,
    any_samples_passed = GL_ANY_SAMPLES_PASSED,
    timestamp          = GL_TIMESTAMP,
};

class Query {
public:
    explicit Query(QueryType type);
    ~Query();

    Query(const Query&) = delete;
    Query& operator=(const Query&) = delete;

    Query(Query&& other) noexcept;
    Query& operator=(Query&& other) noexcept;

    void begin();
    void end();

    void begin_indexed(GLenum target, GLuint index);
    void end_indexed(GLenum target, GLuint index);

    void counter();

    GLuint64 result() const;
    bool result_available() const;

    QueryType type() const { return type_; }
    GLuint handle() const { return handle_; }

private:
    GLuint handle_ = 0;
    QueryType type_;
};

} // namespace gl
