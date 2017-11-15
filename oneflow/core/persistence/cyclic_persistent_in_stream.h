#ifndef ONEFLOW_CORE_PERSISTENCE_CYCLIC_PERSISTENT_IN_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_CYCLIC_PERSISTENT_IN_STREAM_H_

#include "oneflow/core/persistence/persistent_in_stream.h"

namespace oneflow {

class CyclicPersistentInStream final : public PersistentInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CyclicPersistentInStream);
  CyclicPersistentInStream() = delete;
  ~CyclicPersistentInStream() = default;

  CyclicPersistentInStream(fs::FileSystem* fs, const std::string& file_path);

  void AddNForCurFilePos(uint64_t n) override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_CYCLIC_PERSISTENT_IN_STREAM_H_
