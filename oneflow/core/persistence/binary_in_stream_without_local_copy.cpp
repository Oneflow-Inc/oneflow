#include "oneflow/core/persistence/binary_in_stream_without_local_copy.h"
#include "oneflow/core/job/job_desc.h"
#include <cstring>

namespace oneflow {

int32_t BinaryInStreamWithoutLocalCopy::Read(char* s, size_t n) {
  if (IsEof()) return -1;
  CHECK_LE(cur_file_pos_ + n, file_size_);
  file_->Read(cur_file_pos_, n, s);
  cur_file_pos_ += n;
  return 0;
}

BinaryInStreamWithoutLocalCopy::BinaryInStreamWithoutLocalCopy(fs::FileSystem* fs,
                                                               const std::string& file_path,
                                                               uint64_t offset) {
  fs->NewRandomAccessFile(file_path, &file_);
  file_size_ = fs->GetFileSize(file_path);
  cur_file_pos_ = offset;
}

}  // namespace oneflow
