#ifndef ONEFLOW_CORE_PERSISTENCE_NORMAL_PERSISTENT_IN_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_NORMAL_PERSISTENT_IN_STREAM_H_

#include "oneflow/core/persistence/persistent_in_stream.h"

namespace oneflow {

class NormalPersistentInStream final : public PersistentInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalPersistentInStream);
  NormalPersistentInStream() = delete;

  NormalPersistentInStream(fs::FileSystem* fs, const std::string& file_path,
                           uint64_t offset)
      : PersistentInStream(fs, file_path, offset) {
    LOG(INFO) << "New NormalPersistentInStream " << file_path;
  }

  NormalPersistentInStream(fs::FileSystem* fs, const std::string& file_path)
      : NormalPersistentInStream(fs, file_path, 0) {}

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_NORMAL_PERSISTENT_IN_STREAM_H_
