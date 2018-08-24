#ifndef ONEFLOW_CORE_PERSISTENCE_LOG_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_LOG_STREAM_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class LogStream {
 public:
  virtual ~LogStream() = default;
  virtual LogStream& Write(const char* s, size_t n) = 0;
  virtual void Flush() = 0;
};

template<typename T>
typename std::enable_if<std::is_fundamental<T>::value, LogStream&>::type operator<<(
    LogStream& log_stream, const T& x) {
  const char* x_ptr = reinterpret_cast<const char*>(&x);
  size_t n = sizeof(x);
  log_stream.Write(x_ptr, n);
  return log_stream;
}

inline LogStream& operator<<(LogStream& log_stream, const std::string& s) {
  log_stream.Write(s.c_str(), s.size());
  return log_stream;
}

template<size_t n>
LogStream& operator<<(LogStream& log_stream, const char (&s)[n]) {
  log_stream.Write(s, strlen(s));
  return log_stream;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_LOG_STREAM_H_
