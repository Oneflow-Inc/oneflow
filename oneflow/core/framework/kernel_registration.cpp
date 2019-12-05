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
  if (it == registry->end()) { return nullptr; }

  const KernelRegistrationVal* ret = nullptr;
  for (const auto& reg_val : it->second) {
    if (reg_val.is_matched_fn(ctx)) {
      CHECK(ret == nullptr)
          << "There are more than one kernels satisfied by kernel registration context of "
          << op_type_name;
      ret = &reg_val;
    }
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

KernelRegistryWrapperBuilder& KernelRegistryWrapperBuilder::SetIsMatchedPred(
    IsMatchedPredicator fn) {
  wrapper_.reg_val.is_matched_fn = std::move(fn);
  return *this;
}

KernelRegistryWrapperBuilder& KernelRegistryWrapperBuilder::SetInferTmpSizeFn(InferTmpSizeFn fn) {
  wrapper_.reg_val.infer_tmp_size_fn = std::move(fn);
  return *this;
}

KernelRegistryWrapper KernelRegistryWrapperBuilder::Build() {
  CHECK(wrapper_.reg_val.create_fn != nullptr)
      << "No Create function for " << wrapper_.op_type_name;
  CHECK(wrapper_.reg_val.is_matched_fn != nullptr)
      << "No IsMatched function for " << wrapper_.op_type_name;
  if (wrapper_.reg_val.infer_tmp_size_fn == nullptr) {
    wrapper_.reg_val.infer_tmp_size_fn = TmpSizeInferFnUtil::ZeroTmpSize;
  }
  return wrapper_;
}

}  // namespace user_op

}  // namespace oneflow
