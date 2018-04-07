#include "oneflow/core/persistence/cyclic_persistent_in_stream_with_local_copy.h"
#include "oneflow/core/persistence/cyclic_persistent_in_stream_without_local_copy.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"
#include "oneflow/core/common/str_util.h"

namespace oneflow {

CyclicPersistentInStreamWithLocalCopy::CyclicPersistentInStreamWithLocalCopy(
    fs::FileSystem* fs, const std::string& file_path) {
  LOG(INFO) << "New CyclicPersistentInStreamWithLocalCopy " << file_path;
  in_stream_.reset(new NormalPersistentInStream(fs, file_path));
  local_copy_path_ = JoinPath(LogDir(), "global_fs_buffer", file_path);
  out_stream_.reset(new PersistentOutStream(LocalFS(), local_copy_path_));
  read_line_mthd_ =
      &CyclicPersistentInStreamWithLocalCopy::ReadLineAndWriteToLocal;
  read_mthd_ = &CyclicPersistentInStreamWithLocalCopy::ReadAndWriteToLocal;
}

int32_t CyclicPersistentInStreamWithLocalCopy::ReadLineAndWriteToLocal(
    std::string* l) {
  int32_t ret = in_stream_->ReadLine(l);
  if (ret == -1) {
    CopyToLocalFinish();
    return ReadLine(l);
  } else if (ret == 0) {
    *out_stream_ << *l << '\n';
    return 0;
  } else {
    UNIMPLEMENTED();
  }
}

int32_t CyclicPersistentInStreamWithLocalCopy::ReadAndWriteToLocal(char* s,
                                                                   size_t n) {
  int32_t ret = in_stream_->Read(s, n);
  if (ret == -1) {
    CopyToLocalFinish();
    return Read(s, n);
  } else if (ret == 0) {
    out_stream_->Write(s, n);
    return 0;
  } else {
    UNIMPLEMENTED();
  }
}

void CyclicPersistentInStreamWithLocalCopy::CopyToLocalFinish() {
  out_stream_.reset();
  in_stream_.reset(new CyclicPersistentInStreamWithoutLocalCopy(
      LocalFS(), local_copy_path_));
  read_line_mthd_ = &CyclicPersistentInStreamWithLocalCopy::ReadLineFromLocal;
  read_mthd_ = &CyclicPersistentInStreamWithLocalCopy::ReadFromLocal;
}

}  // namespace oneflow
