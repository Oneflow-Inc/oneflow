#ifndef ONEFLOW_CORE_PERSISTENCE_PERSISTENT_OUT_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_PERSISTENT_OUT_STREAM_H_

#include "oneflow/core/common/util.h"
#include "tensorflow/core/platform/env.h"

namespace oneflow {

class PersistentOutStream final {
public:
  OF_DISALLOW_COPY_AND_MOVE(PersistentOutStream);
  PersistentOutStream() = delete;
  ~PersistentOutStream() {
    TF_CHECK_OK(file_->Close());
  }

  PersistentOutStream(const std::string& file_path);

  // Write block of data
  // Inserts the first n characters of the array pointed by s into the stream.
  PersistentOutStream& Write(const char* s, size_t n);

private:
  std::unique_ptr<tensorflow::WritableFile> file_;
};

template<typename T>
PersistentOutStream& operator << (PersistentOutStream& out_stream,
                                  const T& x) {
  static_assert(std::is_fundamental<T>::value, "Not fundamental type");
  const char* x_ptr = &x;
  size_t n = sizeof(x);
  out_stream.Write(x_ptr, n);
  return out_stream;
}

template<>
PersistentOutStream& operator << <std::string>(
    PersistentOutStream& out_stream, const std::string& s);

PersistentOutStream& operator << (
    PersistentOutStream& out_stream, const char* s);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCY_PERSISTENT_OUT_STREAM_H
