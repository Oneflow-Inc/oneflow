#include "oneflow/core/framework/user_op_manager.h"

namespace oneflow {

namespace user_op {

UserOpMgr& UserOpMgr::Get() {
  static UserOpMgr mgr;
  return mgr;
}

OpBuilder UserOpMgr::CheckAndGetOpBuilder(const std::string& op_type_name) {
  CHECK(!op_type_name.empty());
  auto it = op_reg_result_.find(op_type_name);
  CHECK(it == op_reg_result_.end());
  return OpBuilder().Name(op_type_name);
}

void UserOpMgr::Register(OpRegistrationResult& result) {
  CHECK(op_reg_result_.emplace(result.op_type_name, result).seond);
}

const OpRegistrationResult* UserOpMgr::GetOpRegistrationResult(const std::string& op_type_name) {
  auto it = op_reg_result_.find(op_type_name);
  if (it != op_reg_result_.end()) { return &(it->second); }
  return nullptr;
}

OpGradBuilder UserOpMgr::CheckAndGetOpGradBuilder(const std::string& op_type_name) {
  CHECK(!op_type_name.empty());
  auto it = op_grad_reg_result_.find(op_type_name);
  CHECK(it == op_grad_reg_result_.end());
  return OpGradBuilder().Name(op_type_name);
}

void UserOpMgr::Register(OpGradRegistrationResult& result) {
  CHECK(op_grad_reg_result_.emplace(result.op_type_name, result).seond);
}

const OpGradRegistrationResult* UserOpMgr::GetOpGradRegistrationResult(
    const std::string& op_type_name) {
  auto it = op_grad_reg_result_.find(op_type_name);
  if (it != op_grad_reg_result_.end()) { return &(it->second); }
  return nullptr;
}

}  // namespace user_op

}  // namespace oneflow