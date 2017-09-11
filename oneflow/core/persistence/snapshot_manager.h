#ifndef ONEFLOW_CORE_PERSISTENCE_SNAPSHOT_MANAGER_H_
#define ONEFLOW_CORE_PERSISTENCE_SNAPSHOT_MANAGER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/persistence/snapshot.h"

namespace oneflow {

class SnapshotMgr {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SnapshotMgr)
  ~SnapshotMgr() = default;

  OF_SINGLETON(SnapshotMgr);

  Snapshot* GetWriteableSnapshot(int64_t snapshot_id);

  const Snapshot* GetReadableSnapshot() { return readable_snapshot_ptr_.get(); }

  void Init(const Plan& plan);

  size_t num_of_model_blobs() const { return num_of_model_blobs_; }

 private:
  SnapshotMgr() = default;
  std::string model_save_snapshots_path_;
  std::unique_ptr<const Snapshot> readable_snapshot_ptr_;
  HashMap<int64_t, std::unique_ptr<Snapshot>> snapshot_id2writeable_snapshot_;
  size_t num_of_model_blobs_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_SNAPSHOT_MANAGER_H_
