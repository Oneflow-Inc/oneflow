#ifndef ONEFLOW_CORE_PERSISTENCE_LOG_STREAM_SINK_H_
#define ONEFLOW_CORE_PERSISTENCE_LOG_STREAM_SINK_H_

#include <glog/logging.h>
#include <memory>
#include "oneflow/core/persistence/log_stream.h"

namespace oneflow {

class LogStreamSink final : public google::LogSink {
 public:
  LogStreamSink(std::unique_ptr<LogStream>&& out);
  LogStreamSink() = delete;
  ~LogStreamSink() = default;
  void send(google::LogSeverity severity, const char* full_filename, const char* base_filename,
            int line, const struct ::tm* tm_time, const char* message, size_t message_len) override;

 private:
  std::unique_ptr<LogStream> out_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_LOG_STREAM_SINK_H_:
