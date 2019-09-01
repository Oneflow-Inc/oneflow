#ifndef ONEFLOW_CORE_PERSISTENCE_SNAPSHOT_H_
#define ONEFLOW_CORE_PERSISTENCE_SNAPSHOT_H_

#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

class SnapshotReader final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SnapshotReader);
  SnapshotReader() = delete;
  explicit SnapshotReader(const std::string& snapshot_root_path);
  ~SnapshotReader() = default;

  void Read(const std::string& key, Blob* blob) const;
  void Close();

 private:
  const std::string root_path_;
};

class SnapshotWriter final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SnapshotWriter);
  SnapshotWriter() = delete;
  explicit SnapshotWriter(const std::string& snapshot_root_path);
  ~SnapshotWriter() = default;

  void Write(const std::string& key, const Blob* blob);
  void Close();

 private:
  const std::string root_path_;
  std::mutex writer_mutex_;
  bool closed_;
  int64_t writing_count_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_SNAPSHOT_H_
