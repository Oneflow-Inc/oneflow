#ifndef ONEFLOW_CORE_PERSISTENCE_CYCLIC_PERSISTENT_IN_STREAM_WITHOUT_LOCAL_COPY_H_
#define ONEFLOW_CORE_PERSISTENCE_CYCLIC_PERSISTENT_IN_STREAM_WITHOUT_LOCAL_COPY_H_

#include "oneflow/core/persistence/persistent_in_stream_without_local_copy.h"

namespace oneflow {

class CyclicPersistentInStreamWithoutLocalCopy final
    : public PersistentInStreamWithoutLocalCopy {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CyclicPersistentInStreamWithoutLocalCopy);
  CyclicPersistentInStreamWithoutLocalCopy() = delete;
  ~CyclicPersistentInStreamWithoutLocalCopy() = default;

  CyclicPersistentInStreamWithoutLocalCopy(fs::FileSystem* fs,
                                           const std::string& file_path);

 private:
  void UpdateBuffer() override;
  void AddNForCurFilePos(uint64_t n) override;

  bool is_first_update_buffer_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_CYCLIC_PERSISTENT_IN_STREAM_WITHOUT_LOCAL_COPY_H_
