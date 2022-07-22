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
#ifndef ONEFLOW_CORE_FRAMEWORK_TENSOR_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_TENSOR_UTIL_H_

#include <functional>
#include <string>

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/vm/virtual_machine.h"
#include "oneflow/core/common/blocking_then_busy.h"

namespace oneflow {
namespace one {

Maybe<void> SyncAccessTensorWithTimeOut(const std::shared_ptr<Tensor>& tensor,
                                        const std::function<void(uint64_t)>& callback,
                                        const std::string& modifier);

template<typename T>
Maybe<T> GetItem4Tensor(const std::shared_ptr<Tensor>& input) {
  CHECK_EQ_OR_RETURN(input->shape()->elem_cnt(), 1)
      << Error::InvalidValueError() << "only one element tensors can be converted to scalars";
  CHECK_OR_RETURN(IsPODDataType(GetDataType<T>::value))
      << Error::InvalidValueError() << "only POD tensors can be converted to scalars";
  CHECK_OR_RETURN(input->dtype()->data_type() == GetDataType<T>::value)
      << Error::InvalidValueError() << "dtype of input must be same with the template parameters, "
      << kOfBugIssueUploadPrompt;
  T item = -1;
  std::shared_ptr<one::LocalTensor> local_tensor;
  if (input->is_global()) {
    CHECK(NdSbpIsAllBroadcast(*JUST(input->nd_sbp())));
    local_tensor = JUST(input->cur_rank_phy_tensor());
  } else {
    local_tensor = JUST(input->AsLocalTensor());
  }
  const auto& Callback = [&](uint64_t ofblob_ptr) {
    reinterpret_cast<const OfBlob*>(ofblob_ptr)->AutoMemCopyTo<T>(&item, 1);
  };
  auto btb = std::make_shared<BlockingThenBusy>(1);
  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    return builder->SyncAccessBlobByCallback(local_tensor, btb, Callback, "const");
  }));
  JUST(btb->WaitUntilCntEqualZero(VirtualMachine::GetPredicatorNoMoreInstructionsFinished()));
  return item;
}

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_UTIL_H_
