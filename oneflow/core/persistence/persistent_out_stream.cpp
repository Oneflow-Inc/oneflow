#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

PersistentOutStream::PersistentOutStream(fs::FileSystem* fs,
                                         const std::string& file_path) {
  FS_CHECK_OK(fs->NewWritableFile(file_path, &file_));
}

PersistentOutStream::~PersistentOutStream() { FS_CHECK_OK(file_->Close()); }

PersistentOutStream& PersistentOutStream::Write(const char* s, size_t n) {
  FS_CHECK_OK(file_->Append(s, n));
  return *this;
}

}  // namespace oneflow
