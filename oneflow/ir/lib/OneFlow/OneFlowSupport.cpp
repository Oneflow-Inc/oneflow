#include "oneflow/core/framework/user_op_registry_manager.h"

namespace mlir {

namespace oneflow {

namespace support {

using namespace ::oneflow;

const UserOpDef& GetUserOpDef(const std::string& op_type_name) {
  const user_op::OpRegistryResult* val =
      user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_type_name);
  CHECK(val) << " Cannot find op_type_name: " << op_type_name;
  return val->op_def;
}

std::vector<std::string> GetInputKeys(const std::string& op_type_name) {
  std::vector<std::string> ret{};
  for (auto& arg : GetUserOpDef(op_type_name).input()) { ret.push_back(arg.name()); }
  return ret;
}

std::vector<std::string> GetOutputKeys(const std::string& op_type_name) {
  std::vector<std::string> ret{};
  for (auto& arg : GetUserOpDef(op_type_name).output()) { ret.push_back(arg.name()); }
  return ret;
}

}  // namespace support

}  // namespace oneflow

}  // namespace mlir
