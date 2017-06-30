#include "oneflow/core/persistence/persistent_in_stream.h"

namespace oneflow {

PersistentInStream::PersistentInStream(const std::string& file_path,
                                       uint64_t offset) {
  tensorflow::Env* env_ = tensorflow::Env::Default();
  TF_CHECK_OK(env_->FileExists(file_path));
  TF_CHECK_OK(env_->NewRandomAccessFile(file_path, &file_));
  offset_ = offset;
  TF_CHECK_OK(env_->GetFileSize(file_path, &file_size_));
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
  tensorflow::StringPiece result;
  if (file_->Read(offset_, n, &result, s).code() == tensorflow::error::OK) {
    CHECK(result.size() == n);
  } else {
    is_eof_ = true;
  }
  CHECK(result.data() == s);
  offset_ += n;
  return *this;
}

}  // namespace oneflow
