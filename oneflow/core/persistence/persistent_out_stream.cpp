#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

PersistentOutStream::PersistentOutStream(fs::FileSystem* fs,
                                         const std::string& file_path) {
  fs->NewWritableFile(file_path, &file_);
  CHECK(file_.get() != NULL);
}

PersistentOutStream::~PersistentOutStream() { file_->Close(); }

PersistentOutStream& PersistentOutStream::Write(const char* s, size_t n) {
  CHECK(file_->Append(s, n));
  return *this;
}

}  // namespace oneflow
