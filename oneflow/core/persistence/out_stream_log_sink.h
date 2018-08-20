#ifndef ONEFLOW_CORE_PERSISTENCE_OUT_STREAM_LOG_SINK_H_
#define ONEFLOW_CORE_PERSISTENCE_OUT_STREAM_LOG_SINK_H_

#include <glog/logging.h>
#include <memory>
#include "oneflow/core/persistence/out_stream.h"

namespace oneflow {
class OutStreamLogSink : public google::LogSink {
 public:
  OutStreamLogSink(std::unique_ptr<OutStream> out);
  OutStreamLogSink() = delete;
  ~OutStreamLogSink() = default;
  void send(google::LogSeverity severity, const char* full_filename, const char* base_filename,
            int line, const struct ::tm* tm_time, const char* message, size_t message_len);

 private:
  std::unique_ptr<OutStream> out_;
};
}  // namespace oneflow
#endif  // ONEFLOW_CORE_PERSISTENCE_OUT_STREAM_LOG_SINK_H_
