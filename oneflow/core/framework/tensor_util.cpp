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
#include "oneflow/core/framework/tensor_util.h"

#include "oneflow/core/common/blocking_then_busy.h"
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/framework/instructions_builder.h"

namespace oneflow {
namespace one {

Maybe<void> SyncAccessTensorWithTimeOut(const std::shared_ptr<Tensor>& tensor,
                                        const std::function<void(uint64_t)>& Callback,
                                        const std::string& modifier) {
  auto btb = std::make_shared<BlockingThenBusy>(1);
  auto local_tensor = JUST(tensor->AsLocalTensor());
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    return builder->SyncAccessBlobByCallback(local_tensor, btb, Callback, modifier);
  }));
  JUST(btb->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));
  return Maybe<void>::Ok();
}

}  // namespace one
}  // namespace oneflow
