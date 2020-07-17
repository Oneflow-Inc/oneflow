#include "oneflow/core/framework/user_op_grad_builder.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace user_op {

OpGradBuilder& OpGradBuilder::Name(const std::string& op_type_name) {
  CHECK(!op_type_name.empty());
  result_.op_type_name = op_type_name;
}

OpGradBuilder& OpGradBuilder::SetGenBackwardOpConfFn(GenBackwardOpConfFn fn) {
  result_.gen_bw_fn = std::move(fn);
  return *this;
}

OpGradBuilder& OpGradBuilder::Finish() {
  CHECK(result_.gen_bw_fn != nullptr)
      << "No GenBackwardOpConf function for " << result_.op_type_name;
  return *this;
}

OpGradRegistrationResult OpGradBuilder::GetResult() { return result_; }

}  // namespace user_op

}  // namespace oneflow
