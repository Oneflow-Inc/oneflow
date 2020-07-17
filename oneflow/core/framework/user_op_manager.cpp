#include "oneflow/core/framework/user_op_manager.h"

namespace oneflow {

namespace user_op {

UserOpManager& UserOpManager::Get() {
  static UserOpManager mgr;
  return mgr;
}

void UserOpManager::Register(OpBuildResult& result) {
  CHECK(op_info_.emplace(result.op_type_name, result).seond);
}

const OpBuildResult* UserOpManager::GetOpInfo(const std::string& op_type_name) {
  auto it = op_info_.find(op_type_name);
  if (it != op_info_.end()) { return &(it->second); }
  return nullptr;
}

}  // namespace user_op

}  // namespace oneflow