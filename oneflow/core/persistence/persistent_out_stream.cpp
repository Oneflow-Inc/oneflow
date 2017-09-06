#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

PersistentOutStream::PersistentOutStream(fs::FileSystem* file_system,
                                         const std::string& file_path) {
  FS_CHECK_OK(file_system->NewWritableFile(file_path, &file_));
}

PersistentOutStream& PersistentOutStream::Write(const char* s, size_t n) {
  FS_CHECK_OK(file_->Append(s, n));
  return *this;
}

}  // namespace oneflow
