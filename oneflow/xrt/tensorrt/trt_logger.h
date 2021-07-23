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
#ifndef ONEFLOW_XRT_TENSORRT_TRT_LOGGER_H_
#define ONEFLOW_XRT_TENSORRT_TRT_LOGGER_H_

#include <string>
#include "NvInfer.h"

namespace oneflow {
namespace xrt {
namespace tensorrt {

namespace nv {

class Logger : public nvinfer1::ILogger {
 public:
  Logger() = default;

  Logger(const std::string& name) : name_(name) {}

  void log(nvinfer1::ILogger::Severity severity, const char* msg) override;

 private:
  std::string name_ = "TensorRT Logging";
};

}  // namespace nv

}  // namespace tensorrt
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_TENSORRT_TRT_LOGGER_H_
