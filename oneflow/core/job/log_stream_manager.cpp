#include "oneflow/core/persistence/persistent_out_stream.h"
#include "oneflow/core/job/log_stream_manager.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/persistence/tee_persistent_out_stream.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/machine_context.h"

namespace oneflow {

LogStreamMgr::LogStreamMgr() {
  sinks.emplace_back(LocalFS(), LogDir());
  if (SnapshotFS() == LocalFS()) {
  } else {
    sinks.emplace_back(SnapshotFS(),
                       JoinPath(JoinPath(Global<JobDesc>::Get()->MdSaveSnapshotsPath()),
                                GenerateLogDirNameFromContext()));
  }
}

std::unique_ptr<OutStream> LogStreamMgr::Create(const std::string& path) {
  std::vector<std::unique_ptr<PersistentOutStream>> streams;
  for (auto& sink : sinks) {
    streams.emplace_back(
        std::make_unique<PersistentOutStream>(sink.file_system(), JoinPath(sink.prefix(), path)));
  }

  return std::unique_ptr<OutStream>(new TeePersistentOutStream(std::move(streams)));
}

std::string LogStreamMgr::GenerateLogDirNameFromContext() {
  std::ostringstream out;
  out << "log"
      << "." << Global<MachineCtx>::Get()->this_machine_id() << "." << std::time(0);
  return out.str();
}

}  // namespace oneflow
