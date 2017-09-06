#ifndef ONEFLOW_CORE_PERSISTENCE_PERSISTENT_OUT_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_PERSISTENT_OUT_STREAM_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/persistence/file_system.h"

namespace oneflow {

class PersistentOutStream final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PersistentOutStream);
  PersistentOutStream() = delete;
  ~PersistentOutStream();

  PersistentOutStream(fs::FileSystem*, const std::string& file_path);

  // Write block of data
  // Inserts the first n characters of the array pointed by s into the stream.
  PersistentOutStream& Write(const char* s, size_t n);

 private:
  std::unique_ptr<fs::WritableFile> file_;
};

template<typename T>
PersistentOutStream& operator<<(PersistentOutStream& out_stream, const T& x) {
  static_assert(std::is_fundamental<T>::value, "Not fundamental type");
  const char* x_ptr = &x;
  size_t n = sizeof(x);
  out_stream.Write(x_ptr, n);
  return out_stream;
}

inline PersistentOutStream& operator<<(PersistentOutStream& out_stream,
                                       const std::string& s) {
  out_stream.Write(s.c_str(), s.size());
  return out_stream;
}

template<size_t n>
PersistentOutStream& operator<<(PersistentOutStream& out_stream,
                                const char (&s)[n]) {
  out_stream.Write(s, strlen(s));
  return out_stream;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCY_PERSISTENT_OUT_STREAM_H
