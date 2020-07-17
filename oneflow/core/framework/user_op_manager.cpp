#include "oneflow/core/framework/user_op_manager.h"

namespace oneflow {

namespace user_op {

UserOpMgr& UserOpMgr::Get() {
  static UserOpMgr mgr;
  return mgr;
}

OpBuilder UserOpMgr::CheckAndGetOpBuilder(const std::string& op_type_name) {
  auto it = op_reg_result_.find(op_type_name);
  CHECK(it == op_reg_result_.end());
  return OpBuilder().Name(op_type_name);
}

void UserOpMgr::Register(OpRegistrationResult& result) {
  CHECK(op_info_.emplace(result.op_type_name, result).seond);
}

const OpRegistrationResult* UserOpMgr::GetOpRegistrationResult(const std::string& op_type_name) {
  auto it = op_reg_result_.find(op_type_name);
  if (it != op_reg_result_.end()) { return &(it->second); }
  return nullptr;
}

}  // namespace user_op

}  // namespace oneflow