#include "log.hpp"

#include <cstdio>
#include <cstdarg>

namespace gllib {
namespace {

LogCallback s_callback = nullptr;
LogLevel s_min_level = LogLevel::warn;

void stderr_logger(LogLevel level, const char* message) {
    if (level < s_min_level) return;
    const char* prefix = "";
    switch (level) {
        case LogLevel::debug: prefix = "[gllib DEBUG] "; break;
        case LogLevel::info:  prefix = "[gllib INFO]  "; break;
        case LogLevel::warn:  prefix = "[gllib WARN]  "; break;
        case LogLevel::error: prefix = "[gllib ERROR] "; break;
    }
    std::fprintf(stderr, "%s%s\n", prefix, message);
}

} // namespace

void set_log_callback(LogCallback cb) {
    s_callback = cb;
}

LogCallback get_log_callback() {
    return s_callback;
}

void log(LogLevel level, const char* message) {
    if (s_callback) {
        s_callback(level, message);
    }
}

void logf(LogLevel level, const char* fmt, ...) {
    if (!s_callback) return;
    char buf[4096];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    s_callback(level, buf);
}

void log_to_stderr(LogLevel min_level) {
    s_min_level = min_level;
    s_callback = stderr_logger;
}

} // namespace gllib
