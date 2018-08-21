#include "oneflow/core/persistence/persistent_out_stream.h"
#include "oneflow/core/job/log_stream_manager.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/machine_context.h"

namespace oneflow {

LogStreamMgr::LogStreamMgr() {
  destinations_.emplace_back(LocalFS(), LogDir());
  if (SnapshotFS() == LocalFS()) {
  } else {
    destinations_.emplace_back(SnapshotFS(),
                               JoinPath(JoinPath(Global<JobDesc>::Get()->MdSaveSnapshotsPath()),
                                        GenerateLogDirNameFromContext()));
  }
}

std::unique_ptr<LogStream> LogStreamMgr::Create(const std::string& path) const {
  std::vector<std::unique_ptr<PersistentOutStream>> streams;
  for (const auto& destination : destinations_) {
    streams.emplace_back(std::make_unique<PersistentOutStream>(
        destination.mut_file_system(), JoinPath(destination.base_dir(), path)));
  }

  return std::unique_ptr<LogStream>(new TeePersistentLogStream(std::move(streams)));
}

const std::string LogStreamMgr::GenerateLogDirNameFromContext() const {
  std::ostringstream out;
  out << "log"
      << "." << Global<MachineCtx>::Get()->this_machine_id() << "." << std::time(0);
  return out.str();
}

}  // namespace oneflow
