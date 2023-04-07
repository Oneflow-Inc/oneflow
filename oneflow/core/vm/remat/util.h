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

namespace remat {

double append_memory_frag_info_and_get(size_t free_mem, size_t threshold);

Maybe<double> GetComputeTime(const vm::OpCallInstructionPolicy& operand);

}  // namespace remat

namespace vm {

class RematableTensorStorage;
class Stream;
class DtrOpCallInstructionPolicy;

// This class is mainly for holding RematableTensorStorage vector so that we do not
// need to generate them every time.
class RematHelper {
 public:
  explicit RematHelper(const OpCallInstructionPolicy& op_call_instruction_policy);
  RematHelper(const OpCallInstructionPolicy& op_call_instruction_policy, bool inputs_rematable,
              bool outputs_rematable);

  Maybe<void> RematInputs(
      vm::Stream* vm_stream, bool first,
      const std::function<Maybe<void>(OpCallInstructionPolicy*, vm::Stream*)>& compute_fn);
  Maybe<void> EagerlyEvictRemattedTensors(bool first);
  Maybe<void> UpdateRematInfo(bool first, bool recompute, bool include_input, bool include_output);

 private:
  Maybe<int> IncReferenceNumOfRecomputedTensor();
  Maybe<void> _IncReferenceNumOfRecomputedTensor(
      int& pinned_num, std::set<const DtrOpCallInstructionPolicy*>& visited_ops);
  const OpCallInstructionPolicy& op_call_instruction_policy_;
  std::vector<std::shared_ptr<RematableTensorStorage>> input_storages_;
  std::vector<std::shared_ptr<RematableTensorStorage>> output_storages_;
  std::vector<bool> storage_is_initialized_;
};

}  // namespace vm

}  // namespace oneflow
