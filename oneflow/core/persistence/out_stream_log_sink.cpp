#include "oneflow/core/persistence/out_stream_log_sink.h"

namespace oneflow {
OutStreamLogSink::OutStreamLogSink(std::unique_ptr<OutStream> out) : out_(std::move(out)){};

void OutStreamLogSink::send(google::LogSeverity severity, const char* full_filename,
                            const char* base_filename, int line, const struct ::tm* tm_time,
                            const char* message, size_t message_len) {
  (*out_) << ToString(severity, base_filename, line, tm_time, message, message_len) << "\n";
}

}  // namespace oneflow
