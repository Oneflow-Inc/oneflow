#include "oneflow/core/persistence/binary_in_stream_with_local_copy.h"
#include "oneflow/core/persistence/binary_in_stream_without_local_copy.h"
#include "oneflow/core/common/str_util.h"

namespace oneflow {

BinaryInStreamWithLocalCopy::BinaryInStreamWithLocalCopy(fs::FileSystem* fs,
                                                         const std::string& file_path)
    : once_read_(false) {
  LOG(INFO) << "New BinaryInStreamWithLocalCopy " << file_path;
  in_stream_.reset(new BinaryInStreamWithoutLocalCopy(fs, file_path));
  local_copy_path_ = JoinPath(FLAGS_log_dir, "global_fs_buffer", file_path);
  out_stream_.reset(new PersistentOutStream(LocalFS(), local_copy_path_));
  read_mthd_ = &BinaryInStreamWithLocalCopy::ReadAndWriteToLocal;
}

int32_t BinaryInStreamWithLocalCopy::ReadAndWriteToLocal(char* s, size_t n) {
  if (Restart()) {
    CopyToLocalFinish();
    return Read(s, n);
  } else {
    int32_t ret = in_stream_->Read(s, n);
    CHECK_EQ(ret, 0);
    out_stream_->Write(s, n);
    once_read_ = true;
    return 0;
  }
}

bool BinaryInStreamWithLocalCopy::Restart() {
  return in_stream_->cur_file_pos() == 0 && once_read_;
}

void BinaryInStreamWithLocalCopy::CopyToLocalFinish() {
  out_stream_.reset();
  in_stream_.reset(new BinaryInStreamWithoutLocalCopy(LocalFS(), local_copy_path_));
  read_mthd_ = &BinaryInStreamWithLocalCopy::ReadFromLocal;
}

}  // namespace oneflow
