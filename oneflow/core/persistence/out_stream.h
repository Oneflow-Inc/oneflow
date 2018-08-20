#ifndef ONEFLOW_CORE_PERSISTENCE_OUT_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_OUT_STREAM_H_

#include "oneflow/core/common/util.h"

namespace oneflow {
class OutStream {
 public:
  virtual ~OutStream(){};
  virtual OutStream& Write(const char* s, size_t n) = 0;
  virtual void Flush() = 0;
};

template<typename T>
typename std::enable_if<std::is_fundamental<T>::value, OutStream&>::type operator<<(
    OutStream& out_stream, const T& x) {
  const char* x_ptr = reinterpret_cast<const char*>(&x);
  size_t n = sizeof(x);
  out_stream.Write(x_ptr, n);
  return out_stream;
}

inline OutStream& operator<<(OutStream& out_stream, const std::string& s) {
  out_stream.Write(s.c_str(), s.size());
  return out_stream;
}

template<size_t n>
OutStream& operator<<(OutStream& out_stream, const char (&s)[n]) {
  out_stream.Write(s, strlen(s));
  return out_stream;
}
}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_OUT_STREAM_H_
