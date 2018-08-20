#ifndef ONEFLOW_CORE_PERSISTENCE_TEE_PERSISTENT_OUT_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_TEE_PERSISTENT_OUT_STREAM_H_

#include "oneflow/core/persistence/out_stream.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {
class TeePersistentOutStream final : public OutStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TeePersistentOutStream);
  TeePersistentOutStream() = delete;
  TeePersistentOutStream(std::vector<std::unique_ptr<PersistentOutStream>> branches);
  ~TeePersistentOutStream();

  void Flush() override;
  OutStream& Write(const char* s, size_t n) override;

 private:
  std::vector<std::unique_ptr<PersistentOutStream>> branches_;
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_TEE_PERSISTENT_OUT_STREAM_H_
