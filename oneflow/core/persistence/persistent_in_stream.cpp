#include "oneflow/core/persistence/persistent_in_stream.h"

namespace oneflow {

PersistentInStream::PersistentInStream(fs::FileSystem* file_system,
                                       const std::string& file_path,
                                       uint64_t offset) {
  FS_CHECK_OK(file_system->FileExists(file_path));
  FS_CHECK_OK(file_system->NewRandomAccessFile(file_path, &file_));
  offset_ = offset;
  FS_CHECK_OK(file_system->GetFileSize(file_path, &file_size_));
  if (offset < file_size_) {
    is_eof_ = false;
  } else {
    is_eof_ = true;
  }
}

PersistentInStream& PersistentInStream::Read(char* s, size_t n) {
  if (!good()) { return *this; }
  if (offset_ + n > file_size_) {
    is_eof_ = true;
    offset_ += n;
    return *this;
  };
  fs::Status st = file_->Read(offset_, n, s);
  if (st == fs::Status::OUT_OF_RANGE) {
    is_eof_ = true;
  } else if (st != fs::Status::OK) {
    UNEXPECTED_RUN();
  }
  offset_ += n;
  return *this;
}

}  // namespace oneflow
