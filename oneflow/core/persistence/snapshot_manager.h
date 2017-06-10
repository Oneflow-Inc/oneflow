#ifndef ONEFLOW_CORE_PERSISTENCE_SNAPSHOT_MANAGER_H_
#define ONEFLOW_CORE_PERSISTENCE_SNAPSHOT_MANAGER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/persistence/snapshot.h"
#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

class SnapshotManager {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SnapshotManager)
  ~SnapshotManager() = default;

  OF_SINGLETON(SnapshotManager);

  const Snapshot* GetWriterSnapshotFromSnapshotId(uint64_t snapshot_id);

  const Snapshot* GetReadSnapshot() {
    return load_snapshot_ptr_;
  }

  void InitFromPlan(const Plan& plan); 

 private:
  SnapshotManager() = default;
  std::string model_save_snapshots_path_;
  Snapshot* load_snapshot_ptr_;
  HashMap<uint64_t, Snapshot*> snapshot_id2snapshot_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_SNAPSHOT_MANAGER_H_
