#ifndef ONEFLOW_CORE_FRAMEWORK_USER_OP_GRAD_REGISTRY_H_
#define ONEFLOW_CORE_FRAMEWORK_USER_OP_GRAD_REGISTRY_H_

#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace user_op {

using AddOpFn = std::function<void(const UserOpConfWrapper&)>;
using GenBackwardOpConfFn = std::function<void(const UserOpWrapper&, AddOpFn)>;

struct OpGradRegistryResult {
  std::string op_type_name;
  GenBackwardOpConfFn gen_bw_fn;
};

class OpGradRegistry final {
 public:
  OpGradRegistry& Name(const std::string& op_type_name);
  OpGradRegistry& SetGenBackwardOpConfFn(GenBackwardOpConfFn fn);

  OpGradRegistry& Finish();
  OpGradRegistryResult GetResult() { return result_; }

 private:
  OpGradRegistryResult result_;
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_USER_OP_GRAD_REGISTRY_H_
