#include "oneflow/core/persistence/snapshot_manager.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/machine_context.h"
#include <chrono>
#include <ctime>

namespace oneflow {

namespace {

std::string GenNewSnapshotName() {
  const auto now_clock = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now_clock);
  char datetime[sizeof("2006_01_02_15_04_05")];
  std::strftime(datetime, sizeof(datetime), "%Y_%m_%d_%H_%M_%S", std::localtime(&now_time));
  std::ostringstream oss;
  oss << "snapshot_" << datetime << "_" << std::setw(3) << std::setfill('0')
      << std::chrono::duration_cast<std::chrono::milliseconds>(now_clock.time_since_epoch()).count()
             % 1000;
  return oss.str();
}

}  // namespace

SnapshotMgr::SnapshotMgr(const Plan& plan) {
  if (Global<const IOConf>::Get()->enable_write_snapshot()) {
    model_save_snapshots_path_ = Global<const IOConf>::Get()->model_save_snapshots_path();
    CHECK(Global<MachineCtx>::Get()->IsThisMachineMaster());
    SnapshotFS()->CreateDirIfNotExist(model_save_snapshots_path_);
  }
  const std::string& load_path = Global<const IOConf>::Get()->model_load_snapshot_path();
  if (!load_path.empty()) { readable_snapshot_.reset(new Snapshot(load_path)); }
}

Snapshot* SnapshotMgr::GetWriteableSnapshot(int64_t snapshot_id) {
  CHECK(Global<const IOConf>::Get()->enable_write_snapshot());
  std::unique_lock<std::mutex> lck(snapshot_id2writeable_snapshot_mtx_);
  auto it = snapshot_id2writeable_snapshot_.find(snapshot_id);
  if (it == snapshot_id2writeable_snapshot_.end()) {
    const std::string snapshot_root_path =
        JoinPath(model_save_snapshots_path_, GenNewSnapshotName());
    SnapshotFS()->CreateDirIfNotExist(snapshot_root_path);
    std::unique_ptr<Snapshot> ret(new Snapshot(snapshot_root_path));
    auto emplace_ret = snapshot_id2writeable_snapshot_.emplace(snapshot_id, std::move(ret));
    it = emplace_ret.first;
    CHECK(emplace_ret.second);
  }
  return it->second.get();
}

}  // namespace oneflow
