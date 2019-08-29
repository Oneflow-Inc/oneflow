#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/persistence/snapshot_manager.h"

namespace oneflow {

Snapshot::Snapshot(const std::string& snapshot_root_path) : root_path_(snapshot_root_path) {
  CHECK(SnapshotFS()->IsDirectory(snapshot_root_path));
}

std::unique_ptr<PersistentOutStream> Snapshot::GetOutStream(const LogicalBlobId& lbi) {
  const std::string op_name_dir = JoinPath(root_path_, lbi.op_name());
  OfCallOnce(op_name_dir, SnapshotFS(), &fs::FileSystem::CreateDir);
  return std::make_unique<PersistentOutStream>(SnapshotFS(),
                                               JoinPath(op_name_dir, lbi.blob_name()));
}

void Snapshot::Done() {
  const std::string snapshot_done_path = JoinPath(root_path_, "snapshot_done");
  PersistentOutStream out_stream(SnapshotFS(), snapshot_done_path);
}

}  // namespace oneflow
