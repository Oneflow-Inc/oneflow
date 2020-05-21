#include "oneflow/core/framework/grad_registration.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace user_op {

namespace {

// only access with single thread
HashMap<std::string, GradRegistrationVal>* MutGradRegistry() {
  static HashMap<std::string, GradRegistrationVal> registry;
  return &registry;
}

}  // namespace

void GradRegistryWrapper::InsertToGlobalRegistry() {
  CHECK(!op_type_name.empty());
  auto registry = MutGradRegistry();
  CHECK(registry->emplace(op_type_name, reg_val).second);
}

const GradRegistrationVal* LookUpInGradRegistry(const std::string& op_type_name) {
  const auto registry = MutGradRegistry();
  auto it = registry->find(op_type_name);
  if (it != registry->end()) { return &(it->second); }
  return nullptr;
}

std::vector<std::string> GetAllUserOpInGradRegistry() {
  std::vector<std::string> ret;
  const auto registry = MutGradRegistry();
  for (auto it = registry->begin(); it != registry->end(); ++it) { ret.push_back(it->first); }
  return ret;
}

GradRegistryWrapperBuilder::GradRegistryWrapperBuilder(const std::string& op_type_name) {
  wrapper_.op_type_name = op_type_name;
}

GradRegistryWrapperBuilder& GradRegistryWrapperBuilder::SetGenBackwardOpConfFn(
    GenBackwardOpConfFn fn) {
  wrapper_.reg_val.gen_bw_fn = std::move(fn);
  return *this;
}

GradRegistryWrapper GradRegistryWrapperBuilder::Build() {
  CHECK(wrapper_.reg_val.gen_bw_fn != nullptr)
      << "No GenBackwardOpConf function for " << wrapper_.op_type_name;
  return wrapper_;
}

}  // namespace user_op

}  // namespace oneflow
