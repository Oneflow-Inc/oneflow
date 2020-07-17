#include "oneflow/core/framework/user_op_grad_registry.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace user_op {

OpGradRegistry& OpGradRegistry::Name(const std::string& op_type_name) {
  CHECK(!op_type_name.empty());
  result_.op_type_name = op_type_name;
}

OpGradRegistry& OpGradRegistry::SetGenBackwardOpConfFn(GenBackwardOpConfFn fn) {
  result_.gen_bw_fn = std::move(fn);
  return *this;
}

OpGradRegistry& OpGradRegistry::Finish() {
  CHECK(result_.gen_bw_fn != nullptr)
      << "No GenBackwardOpConf function for " << result_.op_type_name;
  return *this;
}

OpGradRegistryResult OpGradRegistry::GetResult() { return result_; }

}  // namespace user_op

}  // namespace oneflow
