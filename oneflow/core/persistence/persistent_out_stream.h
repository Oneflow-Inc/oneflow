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
    if(file_->Close().code() != tensorflow::error::OK) {
      LOG(FATAL) << "can't close file of out_stream";
    }
  }

  PersistentOutStream(const std::string& file_path);

  // the type T must be fundamental
  template<typename T>
  PersistentOutStream& operator << (const T& x) {
    const char* x_ptr = &x;
    static_assert(std::is_fundamental<T>::value, "Not fundamental type");
    size_t n = sizeof(x);
    Write(x_ptr, n);
    return *this;
  }

  // Write block of data
  // Inserts the first n characters of the array pointed by s into the stream.
  PersistentOutStream& Write(const char* s, size_t n);

private:
  std::unique_ptr<tensorflow::WritableFile> file_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCY_PERSISTENT_OUT_STREAM_H
