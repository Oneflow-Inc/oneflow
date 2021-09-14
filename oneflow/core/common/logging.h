#ifndef ONEFLOW_CORE_COMMON_LOGGING_H_
#define ONEFLOW_CORE_COMMON_LOGGING_H_

#include <glog/logging.h>
#include <exception>

namespace google {

namespace base {
namespace internal {
bool GetExitOnDFatal();
void SetExitOnDFatal(bool value);
}  // namespace internal
}  // namespace base

namespace glog_internal_namespace_ {
void DumpStackTraceToString(std::string *stacktrace);
}  // namespace glog_internal_namespace_

class GOOGLE_GLOG_DLL_DECL LogMessageFatalOF : public LogMessage {
 public:
  LogMessageFatalOF(const char* file, int line);
  LogMessageFatalOF(const char* file, int line, const CheckOpString& result);
  __attribute__((noreturn)) ~LogMessageFatalOF() noexcept(false) {
    Flush();
    throw std::exception();
  }
};

#if GOOGLE_STRIP_LOG <= 3
#undef COMPACT_GOOGLE_LOG_FATAL
#define COMPACT_GOOGLE_LOG_FATAL google::LogMessageFatalOF( \
      __FILE__, __LINE__)
#define LOG_TO_STRING_FATAL(message) google::LogMessage( \
      __FILE__, __LINE__, google::GLOG_FATAL, message)
#else
#define COMPACT_GOOGLE_LOG_FATAL google::NullStreamFatal()
#define LOG_TO_STRING_FATAL(message) google::NullStreamFatal()
#endif

#if GOOGLE_STRIP_LOG <= 3
#undef CHECK_OP
#define CHECK_OP(name, op, val1, val2) \
  CHECK_OP_LOG(name, op, val1, val2, google::LogMessageFatalOF)
#else
#define CHECK_OP(name, op, val1, val2) \
  CHECK_OP_LOG(name, op, val1, val2, google::NullStreamFatal)
#endif // STRIP_LOG <= 3

#if (defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L || \
     (defined(_MSC_VER) && _MSC_VER >= 1900))

template <typename T>
T CheckNotNullOF(const char* file, int line, const char* names, T&& t) {
 if (t == nullptr) {
   LogMessageFatalOF(file, line, new std::string(names));
 }
 return std::forward<T>(t);
}

#else

// A small helper for CHECK_NOTNULL().
template <typename T>
T* CheckNotNullOF(const char *file, int line, const char *names, T* t) {
  if (t == NULL) {
    LogMessageFatalOF(file, line, new std::string(names));
  }
  return t;
}
#endif

}  // namespace google

#undef CHECK_NOTNULL
#define CHECK_NOTNULL(val) \
  google::CheckNotNullOF(__FILE__, __LINE__, "'" #val "' Must be non NULL", (val))

#endif // ONEFLOW_CORE_COMMON_LOGGING_H_