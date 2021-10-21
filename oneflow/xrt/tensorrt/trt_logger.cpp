/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
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
