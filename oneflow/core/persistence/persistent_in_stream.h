#ifndef ONEFLOW_CORE_PERSISTENCE_PERSISTENT_IN_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_PERSISTENT_IN_STREAM_H_

#include "oneflow/core/common/util.h"
#include "tensorflow/core/platform/env.h"

namespace oneflow {

class PersistentInStream final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PersistentInStream);
  PersistentInStream() = delete;
  ~PersistentInStream() = default;

  PersistentInStream(const std::string& file_path, uint64_t offset);

  template<typename T>
  PersistentInStream& operator >> (T& x) {
    static_assert(std::is_fundamental<T>::value, "Not fundamental type");
    char* s= &x;
    size_t n = sizeof(x);
    Read(s, n);
    return *this;
  }

  PersistentInStream& Read(char* s, size_t n);

  bool good() const {
    return !is_eof_;
  }

  bool eof() const {
    return is_eof_;
  }

 private:
  std::unique_ptr<tensorflow::RandomAccessFile> file_;
  tensorflow::uint64 file_size_;
  uint64_t offset_;
  bool is_eof_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCY_PERSISTENT_IN_STREAM_H_
