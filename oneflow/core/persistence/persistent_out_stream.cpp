#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

PersistentOutStream::PersistentOutStream(const std::string& file_path) {
  TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(file_path, &file_));
}

PersistentOutStream& PersistentOutStream::Write(const char* s, size_t n) {
  auto data = tensorflow::StringPiece(s, n);
  TF_CHECK_OK(file_->Append(data));
  return *this;
}

template<>
PersistentOutStream& operator << <std::string> (
    PersistentOutStream& out_stream, const std::string& s) {
  out_stream.Write(s.c_str(), s.size());
  return out_stream;
}

PersistentOutStream& operator <<(
    PersistentOutStream& out_stream, const char* s) {
  out_stream.Write(s, strlen(s));
  return out_stream;
}

}  // namespace oneflow
