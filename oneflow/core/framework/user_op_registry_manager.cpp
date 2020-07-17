#include "oneflow/core/framework/user_op_registry_manager.h"

namespace oneflow {

namespace user_op {

UserOpRegistryMgr& UserOpRegistryMgr::Get() {
  static UserOpRegistryMgr mgr;
  return mgr;
}

OpRegistry UserOpRegistryMgr::CheckAndGetOpRegistry(const std::string& op_type_name) {
  CHECK(!op_type_name.empty());
  auto it = op_reg_result_.find(op_type_name);
  CHECK(it == op_reg_result_.end());
  return OpRegistry().Name(op_type_name);
}

void UserOpRegistryMgr::Register(OpRegistryResult& result) {
  CHECK(op_reg_result_.emplace(result.op_type_name, result).seond);
}

const OpRegistryResult* UserOpRegistryMgr::GetOpRegistryResult(const std::string& op_type_name) {
  auto it = op_reg_result_.find(op_type_name);
  if (it != op_reg_result_.end()) { return &(it->second); }
  return nullptr;
}

OpGradRegistry UserOpRegistryMgr::CheckAndGetOpGradRegistry(const std::string& op_type_name) {
  CHECK(!op_type_name.empty());
  auto it = op_grad_reg_result_.find(op_type_name);
  CHECK(it == op_grad_reg_result_.end());
  return OpGradRegistry().Name(op_type_name);
}

void UserOpRegistryMgr::Register(OpGradRegistryResult& result) {
  CHECK(op_grad_reg_result_.emplace(result.op_type_name, result).seond);
}

const OpGradRegistryResult* UserOpRegistryMgr::GetOpGradRegistryResult(
    const std::string& op_type_name) {
  auto it = op_grad_reg_result_.find(op_type_name);
  if (it != op_grad_reg_result_.end()) { return &(it->second); }
  return nullptr;
}

}  // namespace user_op

}  // namespace oneflow