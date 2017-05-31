#ifndef ONEFLOW_RUNTIME_SNAPSHOT_READER_H_
#define ONEFLOW_RUNTIME_SNAPSHOT_READER_H_

#include "common/util.h"

namespace oneflow {

class SnapshotReader final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SnapshotReader);
  SnapshotReader() = delete;
  ~SnapshotReader() = default;

  SnapshotReader(const std::string& snapshot_path);

  void Read(const std::string& key, size_t begin_pos);

 private:

};

} // namespace oneflow

#endif // ONEFLOW_RUNTIME_SNAPSHOT_READER_H_
