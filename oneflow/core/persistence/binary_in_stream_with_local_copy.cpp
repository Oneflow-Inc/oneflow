#include "oneflow/core/persistence/binary_in_stream_with_local_copy.h"
#include "oneflow/core/persistence/binary_in_stream_without_local_copy.h"
#include "oneflow/core/common/str_util.h"

namespace oneflow {

BinaryInStreamWithLocalCopy::BinaryInStreamWithLocalCopy(fs::FileSystem* fs,
                                                         const std::string& file_path) {
  LOG(INFO) << "New BinaryInStreamWithLocalCopy " << file_path;
  in_stream_.reset(new BinaryInStreamWithoutLocalCopy(fs, file_path));
  local_copy_path_ = JoinPath(LogDir(), "global_fs_buffer", file_path);
  out_stream_.reset(new PersistentOutStream(LocalFS(), local_copy_path_));
  read_mthd_ = &BinaryInStreamWithLocalCopy::ReadAndWriteToLocal;
}

int32_t BinaryInStreamWithLocalCopy::ReadAndWriteToLocal(char* s, size_t n) {
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

void BinaryInStreamWithLocalCopy::CopyToLocalFinish() {
  out_stream_.reset();
  in_stream_.reset(new BinaryInStreamWithoutLocalCopy(LocalFS(), local_copy_path_));
  read_mthd_ = &BinaryInStreamWithLocalCopy::ReadFromLocal;
}

}  // namespace oneflow
