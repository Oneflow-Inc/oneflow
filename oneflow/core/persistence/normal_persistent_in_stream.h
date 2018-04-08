#ifndef ONEFLOW_CORE_PERSISTENCE_NORMAL_PERSISTENT_IN_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_NORMAL_PERSISTENT_IN_STREAM_H_

#include "oneflow/core/persistence/persistent_in_stream_without_local_copy.h"

namespace oneflow {

class NormalPersistentInStream final
    : public PersistentInStreamWithoutLocalCopy {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalPersistentInStream);
  NormalPersistentInStream() = delete;

  NormalPersistentInStream(fs::FileSystem* fs, const std::string& file_path,
                           uint64_t offset)
      : PersistentInStreamWithoutLocalCopy(fs, file_path, offset) {
    LOG(INFO) << "New NormalPersistentInStream " << file_path;
  }

  NormalPersistentInStream(fs::FileSystem* fs, const std::string& file_path)
      : NormalPersistentInStream(fs, file_path, 0) {}

 private:
  void AddNForCurFilePos(uint64_t n) override {
    set_cur_file_pos(cur_file_pos() + n);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_NORMAL_PERSISTENT_IN_STREAM_H_
