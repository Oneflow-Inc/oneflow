#include "oneflow/core/persistence/log_stream_sink.h"

namespace oneflow {

LogStreamSink::LogStreamSink(std::unique_ptr<LogStream>&& out) : out_(std::move(out)){};

void LogStreamSink::send(google::LogSeverity severity, const char* full_filename,
                         const char* base_filename, int line, const struct ::tm* tm_time,
                         const char* message, size_t message_len) {
  (*out_) << ToString(severity, base_filename, line, tm_time, message, message_len) << "\n";
}

}  // namespace oneflow
