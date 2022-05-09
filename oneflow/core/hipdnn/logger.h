
#ifdef WITH_HIP

#include <iostream>
#include <sstream>
extern std::ostream cerr;

#if ENABLE_LOG == 0
#define DEBUG_CURRENT_CALL_STACK_LEVEL DEBUG_CALL_STACK_LEVEL_NONE
#endif

#define DEBUG_CALL_STACK_LEVEL_NONE 0
#define DEBUG_CALL_STACK_LEVEL_ERRORS 1
#define DEBUG_CALL_STACK_LEVEL_PROMOTED 2
#define DEBUG_CALL_STACK_LEVEL_INTERNAL_ALLOC 2
#define DEBUG_CALL_STACK_LEVEL_CALLS 3
#define DEBUG_CALL_STACK_LEVEL_MARSHALLING 4
#define DEBUG_CALL_STACK_LEVEL_INFO 5

#ifndef DEBUG_CURRENT_CALL_STACK_LEVEL
#define DEBUG_CURRENT_CALL_STACK_LEVEL DEBUG_CALL_STACK_LEVEL_INFO
#endif

namespace open {

enum class LoggingLevel {
  NONE = 0, // WARNING for Release builds, INFO for Debug builds.
  ERRORS,
  PROMOTED,
  INTERNAL_ALLOC = 2,
  CALLS,
  MARSHALLING = 4,
  INFO = 5
};

int IsLogging(LoggingLevel level);

#define OPEN_LOG(level, ...)                                                   \
  do {                                                                         \
    if (open::IsLogging(level)) {                                              \
      std::cerr << __VA_ARGS__ << std ::endl;                                  \
    }                                                                          \
  } while (false)

#define HIPDNN_OPEN_LOG_E(...) OPEN_LOG(open::LoggingLevel::ERRORS, __VA_ARGS__)
#define HIPDNN_OPEN_LOG_P(...)                                                 \
  OPEN_LOG(open::LoggingLevel::PROMOTED, __VA_ARGS__)
#define HIPDNN_OPEN_LOG_I(...)                                                 \
  OPEN_LOG(open::LoggingLevel::INTERNAL_ALLOC, __VA_ARGS__)
#define HIPDNN_OPEN_LOG_C(...) OPEN_LOG(open::LoggingLevel::CALLS, __VA_ARGS__)
#define HIPDNN_OPEN_LOG_M(...)                                                 \
  OPEN_LOG(open::LoggingLevel::MARSHALLING, __VA_ARGS__)
#define HIPDNN_OPEN_LOG_I2(...) OPEN_LOG(open::LoggingLevel::INFO, __VA_ARGS__)

} // namespace open

#endif //WITH_HIP
