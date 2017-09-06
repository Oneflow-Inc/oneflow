#include "oneflow/core/persistence/persistent_in_stream.h"

namespace oneflow {

PersistentInStream::PersistentInStream(fs::FileSystem* fs,
                                       const std::string& file_path,
                                       uint64_t offset) {
  FS_CHECK_OK(fs->FileExists(file_path));
  FS_CHECK_OK(fs->NewRandomAccessFile(file_path, &file_));
  offset_ = offset;
  FS_CHECK_OK(fs->GetFileSize(file_path, &file_size_));
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
  }
  FS_CHECK_OK(file_->Read(offset_, n, s));
  offset_ += n;
  return *this;
}

}  // namespace oneflow
