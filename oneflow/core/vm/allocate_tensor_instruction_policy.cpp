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
#include "oneflow/core/vm/allocate_tensor_instruction_policy.h"

namespace oneflow {
namespace vm {

AllocateTensorInstructionPolicy::AllocateTensorInstructionPolicy(
    const EagerBlobObjectList& eager_blob_objects, vm::Stream* vm_stream)
    : eager_blob_objects_(eager_blob_objects) {
  stream_sequential_dependence_ = vm_stream->schedule_local_dep_object().get();
  for (const auto& eager_blob_object : eager_blob_objects) {
    output_dependences_.push_back(CHECK_JUST(eager_blob_object->compute_local_dep_object()));
  }
}

std::string AllocateTensorInstructionPolicy::DebugName(const vm::Instruction& instruction) const {
  return "AllocateTensor";
}

void AllocateTensorInstructionPolicy::Compute(Instruction* instruction) {
  Allocator* allocator = instruction->mut_stream()->mut_stream_policy()->mut_allocator();
  for (const auto& eager_blob_object : eager_blob_objects_) {
    CHECK_JUST(eager_blob_object->TryAllocateBlobBodyMemory(allocator));
  }
}

}  // namespace vm
}  // namespace oneflow
