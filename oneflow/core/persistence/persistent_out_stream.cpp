#include "oneflow/core/persistence/persistent_out_stream.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/machine_context.h"

namespace oneflow {

PersistentOutStream::PersistentOutStream(fs::FileSystem* fs, const std::string& file_path) {
  std::string file_dir = Dirname(file_path);
  OfCallOnce(Global<MachineCtx>::Get()->GetThisCtrlAddr() + ":"
                 + Global<MachineCtx>::Get()->GetThisCtrlPort() + "/" + file_dir,
             fs, &fs::FileSystem::RecursivelyCreateDirIfNotExist, file_dir);
  fs->NewWritableFile(file_path, &file_);
}

PersistentOutStream::~PersistentOutStream() { file_->Close(); }

PersistentOutStream& PersistentOutStream::Write(const char* s, size_t n) {
  file_->Append(s, n);
  return *this;
}

void PersistentOutStream::Flush() { file_->Flush(); }

}  // namespace oneflow
