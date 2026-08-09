#pragma once

namespace gllib {

enum class LogLevel {
    debug = 0,
    info  = 1,
    warn  = 2,
    error = 3,
};

using LogCallback = void (*)(LogLevel level, const char* message);

void set_log_callback(LogCallback cb);
LogCallback get_log_callback();

void log(LogLevel level, const char* message);
void logf(LogLevel level, const char* fmt, ...);

void log_to_stderr(LogLevel min_level = LogLevel::warn);

} // namespace gllib
