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
#ifndef ONEFLOW_XRT_OPENVINO_OPENVINO_EXECUTABLE_H_
#define ONEFLOW_XRT_OPENVINO_OPENVINO_EXECUTABLE_H_

#include <inference_engine.hpp>

#include "oneflow/xrt/executable.h"
#include "oneflow/xrt/parameter.h"
#include "oneflow/xrt/openvino/inference_engine_data_desc.h"
#include "oneflow/xrt/utility/stl.h"

namespace oneflow {
namespace xrt {
namespace openvino {

class OpenvinoExecutable : public Executable {
 public:
  OpenvinoExecutable(std::unique_ptr<InferenceEngine::ExecutableNetwork> network,
                     const util::Map<std::string, int>& in_out_to_param_idx)
      : Executable("", XrtEngine::OPENVINO),
        executable_network_(std::move(network)),
        in_out_to_param_idx_(in_out_to_param_idx) {}
  virtual ~OpenvinoExecutable() = default;

  bool Run(const std::vector<Parameter>& inputs, const ExecutableRunOptions& run_options,
           bool block_until_done = true) override;

  InferenceEngine::Blob::Ptr ParameterToBlobPtr(const Parameter& input,
                                                const InferenceEngine::TensorDesc& in_desc);

 private:
  std::unique_ptr<InferenceEngine::ExecutableNetwork> executable_network_;
  util::Map<std::string, int> in_out_to_param_idx_;
};

}  // namespace openvino
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_OPENVINO_OPENVINO_EXECUTABLE_H_
