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
#ifndef ONEFLOW_XRT_TVM_EXECUTABLE_H_
#define ONEFLOW_XRT_TVM_EXECUTABLE_H_

#include "oneflow/xrt/executable.h"
#include "oneflow/xrt/parameter.h"
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>

namespace oneflow {
namespace xrt {
namespace of_tvm {

class TVMExecutable final : public Executable {
 public:
  TVMExecutable(const std::string& name, const int num_inputs,
                const std::vector<Parameter>& outputs, const std::string& json,
                const tvm::runtime::Module& built_mod, XrtDevice device);

  bool Run(const std::vector<Parameter>& inputs, const ExecutableRunOptions& run_options,
           bool block_until_done = true) override;

 private:
  int num_inputs_;
  std::vector<Parameter> outputs_;
  tvm::runtime::Module built_mod_;
  std::string graph_json_;
  XrtDevice device_;

  bool is_inited_;
  TVMContext ctx_;
  std::vector<DLManagedTensor> output_dltensors_;
  tvm::runtime::Module executor_;
  tvm::runtime::PackedFunc set_input_;
  tvm::runtime::PackedFunc run_;
  tvm::runtime::PackedFunc get_output_;
  tvm::runtime::PackedFunc get_num_outputs_;
};

}  // namespace of_tvm
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_TVM_EXECUTABLE_H_
