#include "oneflow/core/framework/kernel_registration.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/kernel/kernel.pb.h"
#include "oneflow/core/framework/tensor_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/framework/infer_util.h"

namespace oneflow {

namespace user_op {

namespace {

// only access with single thread
HashMap<std::string, std::vector<KernelRegistrationVal>>* MutKernelRegistry() {
  static HashMap<std::string, std::vector<KernelRegistrationVal>> registry;
  return &registry;
}

std::string GetErrorMsgOfSearchedOp(const KernelRegContext& ctx) {
  const auto& op_conf = ctx.user_op_conf();
  std::stringstream ss;
  ss << " The info of Op node are "
     << "\n op_name: " << op_conf.op_name() << "\n op_type_name: " << op_conf.op_type_name()
     << "\n DeviceType_Name: " << DeviceType_Name(ctx.device_type());
  for (const auto& pair : ctx.inputs()) {
    ss << "\n DataType_Name of " << pair.first << "_" << pair.second << ": "
       << DataType_Name(ctx.TensorDesc4ArgNameAndIndex(pair.first, pair.second)->data_type());
  }
  for (const auto& pair : ctx.outputs()) {
    ss << "\n DataType_Name of " << pair.first << "_" << pair.second << ": "
       << DataType_Name(ctx.TensorDesc4ArgNameAndIndex(pair.first, pair.second)->data_type());
  }
  return ss.str();
}

}  // namespace

void KernelRegistryWrapper::InsertToGlobalRegistry() {
  CHECK(!op_type_name.empty());
  auto registry = MutKernelRegistry();
  (*registry)[op_type_name].emplace_back(reg_val);
}

const KernelRegistrationVal* LookUpInKernelRegistry(const std::string& op_type_name,
                                                    const KernelRegContext& ctx) {
  const auto registry = MutKernelRegistry();
  auto it = registry->find(op_type_name);
  if (it == registry->end()) {
    LOG(ERROR) << "There is no kernel registered for current Op node. "
               << GetErrorMsgOfSearchedOp(ctx);
    return nullptr;
  }

  const KernelRegistrationVal* ret = nullptr;
  for (const auto& reg_val : it->second) {
    if (reg_val.is_matched_hob(ctx)) {
      CHECK(ret == nullptr) << "There are more than one kernels satisfy current Op node. "
                            << GetErrorMsgOfSearchedOp(ctx);
      ret = &reg_val;
    }
  }
  if (ret == nullptr) {
    LOG(ERROR) << "Cannot find the kernel satisfies current Op node. "
               << GetErrorMsgOfSearchedOp(ctx);
  }

  return ret;
}

std::vector<std::string> GetAllUserOpInKernelRegistry() {
  std::vector<std::string> ret;
  const auto registry = MutKernelRegistry();
  for (auto it = registry->begin(); it != registry->end(); ++it) { ret.push_back(it->first); }
  return ret;
}

KernelRegistryWrapperBuilder::KernelRegistryWrapperBuilder(const std::string& op_type_name) {
  wrapper_.op_type_name = op_type_name;
}

KernelRegistryWrapperBuilder& KernelRegistryWrapperBuilder::SetCreateFn(CreateFn fn) {
  wrapper_.reg_val.create_fn = std::move(fn);
  return *this;
}

KernelRegistryWrapperBuilder& KernelRegistryWrapperBuilder::SetIsMatchedHob(IsMatchedHob hob) {
  wrapper_.reg_val.is_matched_hob = hob;
  return *this;
}

KernelRegistryWrapperBuilder& KernelRegistryWrapperBuilder::SetInferTmpSizeFn(InferTmpSizeFn fn) {
  wrapper_.reg_val.infer_tmp_size_fn = std::move(fn);
  return *this;
}

KernelRegistryWrapperBuilder& KernelRegistryWrapperBuilder::SetInplaceProposalFn(
    InplaceProposalFn fn) {
  wrapper_.reg_val.inplace_proposal_fn = std::move(fn);
  return *this;
}

KernelRegistryWrapper KernelRegistryWrapperBuilder::Build() {
  CHECK(wrapper_.reg_val.create_fn != nullptr)
      << "No Create function for " << wrapper_.op_type_name;
  if (wrapper_.reg_val.infer_tmp_size_fn == nullptr) {
    wrapper_.reg_val.infer_tmp_size_fn = TmpSizeInferFnUtil::ZeroTmpSize;
  }
  if (wrapper_.reg_val.inplace_proposal_fn == nullptr) {
    wrapper_.reg_val.inplace_proposal_fn = [](const InferContext&, AddInplaceArgPair) {
      return Maybe<void>::Ok();
    };
  }
  return wrapper_;
}

}  // namespace user_op

}  // namespace oneflow
