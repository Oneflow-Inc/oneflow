#include "oneflow/core/framework/user_op_kernel_registry.h"

namespace oneflow {

namespace user_op {

OpKernelRegistry& OpKernelRegistry::Name(const std::string& op_type_name) {
  result_.op_type_name = op_type_name;
  return *this;
}

OpKernelRegistry& OpKernelRegistry::SetCreateFn(CreateFn fn) {
  result_.create_fn = std::move(fn);
  return *this;
}

OpKernelRegistry& OpKernelRegistry::SetIsMatchedHob(IsMatchedHob hob) {
  result_.is_matched_hob = hob;
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

OpKernelRegistry& OpKernelRegistry::Finish() {
  CHECK(result_.create_fn != nullptr) << "No Create function for " << result_.op_type_name;
  if (result_.infer_tmp_size_fn == nullptr) {
    result_.infer_tmp_size_fn = TmpSizeInferFnUtil::ZeroTmpSize;
  }
  if (result_.inplace_proposal_fn == nullptr) {
    result_.inplace_proposal_fn = [](const InferContext&, AddInplaceArgPair) {
      return Maybe<void>::Ok();
    };
  }
  return *this;
}

}  // namespace user_op

}  // namespace oneflow
