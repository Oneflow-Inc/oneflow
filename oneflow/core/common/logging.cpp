#include "oneflow/core/common/logging.h"
#include "oneflow/core/common/util.h"
#include <stdexcept>

namespace google {
COMMAND(base::internal::SetExitOnDFatal(true));

LogMessageFatalOF::LogMessageFatalOF(const char* file, int line) :
    LogMessage(file, line, GLOG_FATAL) {}

LogMessageFatalOF::LogMessageFatalOF(const char* file, int line,
                                 const CheckOpString& result) :
    LogMessage(file, line, result) {}

}  // namespace google
