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
#ifndef ONEFLOW_XRT_EXECUTABLE_H_
#define ONEFLOW_XRT_EXECUTABLE_H_

#include <vector>

#include "oneflow/xrt/parameter.h"
#include "oneflow/xrt/xrt.pb.h"

namespace oneflow {
namespace xrt {

struct ExecutableRunOptions {
  // Specify stream if the engine supports multiple computation streams.
  // It will use the default computation stream if `stream` is not set.
  void* stream = nullptr;

  int32_t device_ordinal = -1;

  // Set host threads num.
  int32_t host_num_threads = -1;

  // Limit memory footprint.
  int64_t host_memory_limit = -1;
  int64_t device_memory_limit = -1;

  // Random seed.
  int64_t random_seed = -1;

  // Maximum batch size for TensorRT.
  int32_t max_batch_size = 1;

  // Enable TensorRT Mixed-Precision.
  // Enable TensorRT fp16
  bool tensorrt_fp16 = false;
  // Enable TensorRT int8
  bool tensorrt_int8 = false;

  std::string tensorrt_int8_calibration = "";

  // Feed the return parameters to reuse it's storage while running
  // the executable.
  std::vector<Parameter> return_params;
};

class Executable {
 public:
  Executable(const std::string& name, const XrtEngine& engine)  // NOLINT
      : name_(name), engine_(engine) {}
  virtual ~Executable() = default;

  const XrtEngine& engine() const { return engine_; }

  const std::string& name() const { return name_; }

  virtual bool Run(const std::vector<Parameter>& inputs,     // NOLINT
                   const ExecutableRunOptions& run_options,  // NOLINT
                   bool block_until_done = true) = 0;

  bool RunAsync(const std::vector<Parameter> inputs,  // NOLINT
                const ExecutableRunOptions& run_options) {
    return Run(inputs, run_options, false);
  }

  const std::vector<Parameter>& Results() const { return results_; }

 protected:
  // Executable name.
  std::string name_;
  // Executable engine, XLA or TensorRT.
  XrtEngine engine_;
  std::vector<Parameter> results_;
};

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_EXECUTABLE_H_
