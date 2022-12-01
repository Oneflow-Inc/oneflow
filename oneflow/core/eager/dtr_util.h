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
#include <memory>
#include <vector>

#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/env_var/dtr.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/vm/dtr_ep_allocator.h"

namespace oneflow {

namespace vm {
class DtrEpAllocator;
}

namespace dtr {

bool is_enabled();
size_t memory_threshold();
bool is_enabled_and_debug();
int debug_level();
bool is_check_enabled();
double append_memory_frag_info_and_get(size_t free_mem, size_t threshold);

class AllocatorManager {
 public:
  vm::DtrEpAllocator* CreateOrGetAllocator(DeviceType device_type, size_t device_index);
 private:
  std::unordered_map<std::pair<DeviceType, size_t>, std::unique_ptr<vm::DtrEpAllocator>> allocators_;
};

}  // namespace dtr

// namespace vm {
//
// std::vector<std::shared_ptr<DTREagerBlobObject>> GetDTRInputs(
//     const LocalCallOpKernelPhyInstrOperand* operand);
// std::vector<std::shared_ptr<DTREagerBlobObject>> GetDTROutputs(
//     const LocalCallOpKernelPhyInstrOperand* operand);
//
// std::vector<std::shared_ptr<DTREagerBlobObject>> GetDTRInputs(
//     const std::shared_ptr<const LocalCallOpKernelPhyInstrOperand>& operand);
// std::vector<std::shared_ptr<DTREagerBlobObject>> GetDTROutputs(
//     const std::shared_ptr<const LocalCallOpKernelPhyInstrOperand>& operand);
//
// std::shared_ptr<LocalCallOpKernelPhyInstrOperand> DTROp2LocalCallOp(DTRInstrOperand* operand);
//
// Maybe<void> CheckInputInMemory(LocalCallOpKernelPhyInstrOperand* operand);
// Maybe<void> CheckOutputInMemory(LocalCallOpKernelPhyInstrOperand* operand);
//
// }  // namespace vm
}  // namespace oneflow
