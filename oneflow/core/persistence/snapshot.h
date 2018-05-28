#ifndef ONEFLOW_CORE_PERSISTENCE_SNAPSHOT_H_
#define ONEFLOW_CORE_PERSISTENCE_SNAPSHOT_H_

#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

class Snapshot final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Snapshot);
  Snapshot() = delete;
  ~Snapshot() = default;

  Snapshot(const std::string& snapshot_root_path);

  std::unique_ptr<PersistentOutStream> GetOutStream(const LogicalBlobId& lbi, int32_t part_id);

  void OnePartDone(const LogicalBlobId& lbi, int32_t part_id, int32_t part_num);

  std::string GetDirFromOpName(const std::string& op_name) const;

 private:
  void ConcatLbnFile(const LogicalBlobId& lbi, int32_t part_num, const std::string& concat_file);

  std::string root_path_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_SNAPSHOT_H_
