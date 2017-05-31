#ifndef ONEFLOW_RUNTIME_SNAPSHOT_WRITER_H_
#define ONEFLOW_RUNTIME_SNAPSHOT_WRITER_H_

#include "common/util.h"

namespace oneflow {

class SnapshotWriter final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SnapshotWriter);
  SnapshotWriter() = delete;
  ~SnapshotWriter() = default;

  SnapshotWriter(const std::string& snapshot_path);

  void PrepareWrite(const std::string& key, int32_t part_num);
  void Write(const std::string& key, int32_t part_id);

 private:

};

} // namespace oneflow

#endif // ONEFLOW_RUNTIME_SNAPSHOT_WRITER_H_
