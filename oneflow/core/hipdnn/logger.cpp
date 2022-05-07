
#ifdef WITH_HIP

#include "logger.h"
#include <sstream>

namespace open {

int IsLogging(const LoggingLevel level) {
  const int enable_level = DEBUG_CURRENT_CALL_STACK_LEVEL;
  return enable_level >= ((int)level - 1);
}
} // namespace open

#endif //WITH_HIP
