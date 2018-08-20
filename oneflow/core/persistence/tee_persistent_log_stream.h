#ifndef ONEFLOW_CORE_PERSISTENCE_TEE_PERSISTENT_LOG_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_TEE_PERSISTENT_LOG_STREAM_H_

#include "oneflow/core/persistence/log_stream.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

class TeePersistentLogStream final : public LogStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TeePersistentLogStream);
  TeePersistentLogStream() = delete;
  explicit TeePersistentLogStream(std::vector<std::unique_ptr<PersistentOutStream>>&& branches);
  ~TeePersistentLogStream();

  void Flush() override;
  LogStream& Write(const char* s, size_t n) override;

 private:
  std::vector<std::unique_ptr<PersistentOutStream>> branches_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_TEE_PERSISTENT_LOG_STREAM_H_
