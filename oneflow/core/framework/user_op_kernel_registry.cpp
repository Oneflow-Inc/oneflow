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
#include "oneflow/core/framework/user_op_kernel_registry.h"
#include "oneflow/core/framework/user_op_hob.h"

namespace oneflow {

namespace user_op {

OpKernelRegistry& OpKernelRegistry::Name(const std::string& op_type_name) {
  result_.op_type_name = op_type_name;
  return *this;
}

OpKernelRegistry& OpKernelRegistry::SetCreateFn(OpKernelCreateFn fn) {
  result_.create_fn = std::move(fn);
  return *this;
}

OpKernelRegistry& OpKernelRegistry::SetInferTmpSizeFn(InferTmpSizeFn fn) {
  result_.infer_tmp_size_fn = std::move(fn);
  return *this;
}

OpKernelRegistry& OpKernelRegistry::SetInplaceProposalFn(InplaceProposalFn fn) {
  result_.inplace_proposal_fn = std::move(fn);
  return *this;
}

OpKernelRegistry& OpKernelRegistry::SetPriority(int32_t priority) {
  result_.priority = priority;
  return *this;
}

Maybe<OpKernelRegistry&> OpKernelRegistry::Finish() {
  CHECK_OR_RETURN(result_.create_fn != nullptr)
      << "No Create function for " << result_.op_type_name;
  result_.need_temp_storage = (result_.infer_tmp_size_fn != nullptr);
  if (!result_.need_temp_storage) { result_.infer_tmp_size_fn = TmpSizeInferFnUtil::ZeroTmpSize; }
  if (result_.inplace_proposal_fn == nullptr) {
    result_.inplace_proposal_fn = [](const InferContext&, AddInplaceArgPair) {
      return Maybe<void>::Ok();
    };
  }
  if (result_.is_matched_hob == nullptr) {
    static auto hob_true = std::make_shared<decltype(user_op::HobTrue())>(user_op::HobTrue());
    result_.is_matched_hob = hob_true;
  }
  return *this;
}

}  // namespace user_op

}  // namespace oneflow
