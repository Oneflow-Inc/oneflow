#ifndef ONEFLOW_CORE_FRAMEWORK_GRAD_REGISTRATION_H_
#define ONEFLOW_CORE_FRAMEWORK_GRAD_REGISTRATION_H_

#include "oneflow/core/framework/registrar.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace user_op {

using AddOpFn = std::function<void(const UserOpConfWrapper&)>;
using GenBackwardOpConfFn = std::function<void(const UserOpWrapper&, AddOpFn)>;

struct GradRegistrationVal {
  GenBackwardOpConfFn gen_bw_fn;
};

struct GradRegistryWrapper final {
  void InsertToGlobalRegistry();

  std::string op_type_name;
  GradRegistrationVal reg_val;
};

class GradRegistryWrapperBuilder final {
 public:
  GradRegistryWrapperBuilder(const std::string& op_type_name);
  GradRegistryWrapperBuilder& SetGenBackwardOpConfFn(GenBackwardOpConfFn fn);

  GradRegistryWrapper Build();

 private:
  GradRegistryWrapper wrapper_;
};

const GradRegistrationVal* LookUpInGradRegistry(const std::string& op_type_name);

std::vector<std::string> GetAllUserOpInGradRegistry();

}  // namespace user_op

}  // namespace oneflow

#define REGISTER_USER_OP_GRAD(name)                                                               \
  static ::oneflow::user_op::Registrar<::oneflow::user_op::GradRegistryWrapperBuilder> OF_PP_CAT( \
      g_registrar, __COUNTER__) = ::oneflow::user_op::GradRegistryWrapperBuilder(name)

#endif  // ONEFLOW_CORE_FRAMEWORK_GRAD_REGISTRATION_H_
