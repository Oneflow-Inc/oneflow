#ifndef ONEFLOW_CORE_FRAMEWORK_USER_OP_GRAD_BUILDER_H_
#define ONEFLOW_CORE_FRAMEWORK_USER_OP_GRAD_BUILDER_H_

#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace user_op {

using AddOpFn = std::function<void(const UserOpConfWrapper&)>;
using GenBackwardOpConfFn = std::function<void(const UserOpWrapper&, AddOpFn)>;

struct OpGradRegistrationResult {
  std::string op_type_name;
  GenBackwardOpConfFn gen_bw_fn;
};

class OpGradBuilder final {
 public:
  OpGradBuilder& Name(const std::string& op_type_name);
  OpGradBuilder& SetGenBackwardOpConfFn(GenBackwardOpConfFn fn);
  OpGradBuilder& Finish();
  OpGradRegistrationResult GetResult();

 private:
  OpGradRegistrationResult result_;
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_USER_OP_GRAD_BUILDER_H_
