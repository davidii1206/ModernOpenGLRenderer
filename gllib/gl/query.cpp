#include "query.hpp"

#include <gllib/log.hpp>

namespace gl {

Query::Query(QueryType type)
    : type_(type)
{
    if (type == QueryType::timestamp) {
        glGenQueries(1, &handle_);
    } else {
        glCreateQueries(static_cast<GLenum>(type), 1, &handle_);
    }
}

Query::~Query() {
    if (handle_) {
        glDeleteQueries(1, &handle_);
    }
}

Query::Query(Query&& other) noexcept
    : handle_(other.handle_)
    , type_(other.type_)
{
    other.handle_ = 0;
}

Query& Query::operator=(Query&& other) noexcept {
    if (this != &other) {
        if (handle_) glDeleteQueries(1, &handle_);
        handle_ = other.handle_;
        type_ = other.type_;
        other.handle_ = 0;
    }
    return *this;
}

void Query::begin() {
    glBeginQuery(static_cast<GLenum>(type_), handle_);
}

void Query::end() {
    glEndQuery(static_cast<GLenum>(type_));
}

void Query::begin_indexed(GLenum target, GLuint index) {
    glBeginQueryIndexed(target, index, handle_);
}

void Query::end_indexed(GLenum target, GLuint index) {
    glEndQueryIndexed(target, index);
}

void Query::counter() {
    glQueryCounter(handle_, GL_TIMESTAMP);
}

GLuint64 Query::result() const {
    GLuint64 val = 0;
    glGetQueryObjectui64v(handle_, GL_QUERY_RESULT, &val);
    return val;
}

bool Query::result_available() const {
    GLint available = 0;
    glGetQueryObjectiv(handle_, GL_QUERY_RESULT_AVAILABLE, &available);
    return available != 0;
}

} // namespace gl
