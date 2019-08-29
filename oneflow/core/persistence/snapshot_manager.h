#ifndef ONEFLOW_CORE_PERSISTENCE_SNAPSHOT_MANAGER_H_
#define ONEFLOW_CORE_PERSISTENCE_SNAPSHOT_MANAGER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/persistence/snapshot.h"

namespace oneflow {

class SnapshotMgr {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SnapshotMgr)
  SnapshotMgr() = delete;
  ~SnapshotMgr() = default;

  Snapshot* GetWriteableSnapshot(int64_t snapshot_id);
  const Snapshot* GetReadableSnapshot() { return readable_snapshot_.get(); }

 private:
  friend class Global<SnapshotMgr>;
  SnapshotMgr(const Plan& plan);

  std::string model_save_snapshots_path_;
  std::unique_ptr<const Snapshot> readable_snapshot_;
  std::mutex snapshot_id2writeable_snapshot_mtx_;
  HashMap<int64_t, std::unique_ptr<Snapshot>> snapshot_id2writeable_snapshot_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_SNAPSHOT_MANAGER_H_
