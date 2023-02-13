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
#pragma once

#include <memory>

#include "oneflow/core/common/maybe.h"

namespace oneflow {

namespace vm {
class OpCallInstructionPolicy;
}

namespace dtr {

double append_memory_frag_info_and_get(size_t free_mem, size_t threshold);

Maybe<double> GetComputeTime(const vm::OpCallInstructionPolicy& operand);

}  // namespace dtr

namespace vm {

class RematableTensorStorage;
class Stream;

class Pack {
 public:
  const OpCallInstructionPolicy& op_call_instruction_policy;
  std::vector<std::shared_ptr<RematableTensorStorage>> input_storages;
  std::vector<std::shared_ptr<RematableTensorStorage>> output_storages;
  explicit Pack(const OpCallInstructionPolicy& op_call_instruction_policy);
};

Maybe<void> RematInputs(
    const Pack& pack, vm::Stream* vm_stream, bool first,
    const std::function<Maybe<void>(OpCallInstructionPolicy*, vm::Stream*)>& compute_fn);
Maybe<void> EagerlyEvictRemattedTensors(const Pack& pack, vm::Stream* vm_stream, bool first);
Maybe<void> UpdateRematInfo(const Pack& pack, vm::Stream* vm_stream, bool first, bool recompute,
                            bool include_input, bool include_output,
                            const std::vector<bool>& storage_is_initialized);
}  // namespace vm

}  // namespace oneflow
