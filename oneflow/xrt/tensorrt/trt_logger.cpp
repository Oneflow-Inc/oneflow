#include "oneflow/xrt/tensorrt/trt_logger.h"
#include "glog/logging.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

namespace nv {

using ILogger = ::nvinfer1::ILogger;

void Logger::log(ILogger::Severity severity, const char* msg) {
  switch (severity) {
    case ILogger::Severity::kVERBOSE:
    case ILogger::Severity::kINFO: {
      LOG(INFO) << name_ << ": " << msg;
      break;
    }
    case ILogger::Severity::kWARNING: {
      LOG(WARNING) << name_ << ": " << msg;
      break;
    }
    case ILogger::Severity::kERROR: {
      LOG(FATAL) << name_ << ": " << msg;
      break;
    }
    case ILogger::Severity::kINTERNAL_ERROR: {
      LOG(FATAL) << name_ << ": " << msg;
      break;
    }
    default: {
      LOG(FATAL) << name_ << ": Unknow severity level " << int(severity)
                 << " with message: " << msg;
      break;
    }
  }
}

}  // namespace nv

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow
