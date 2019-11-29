#ifndef ONEFLOW_CORE_FRAMEWORK_USER_OP_DEF_WRAPPER_H_
#define ONEFLOW_CORE_FRAMEWORK_USER_OP_DEF_WRAPPER_H_

#include "oneflow/core/framework/user_op_def.pb.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace user_op {

class UserOpDefWrapper final {
 public:
  UserOpDefWrapper(const UserOpDef&);
  ~UserOpDefWrapper() = default;
  UserOpDefWrapper(const UserOpDefWrapper&) = delete;
  UserOpDefWrapper(UserOpDefWrapper&&) = delete;

  const std::string& name() const { return def_.name(); }

  bool IsInputArgName(const std::string&) const;
  bool IsOutputArgName(const std::string&) const;
  bool IsAttrName(const std::string&) const;

  bool IsArgOptional(const std::string&) const;
  std::pair<int32_t, bool> ArgNumAndIsMin(const std::string&) const;

  UserOpAttrType GetAttrType(const std::string&) const;
  bool AttrHasDefaultVal(const std::string&) const;
  template<typename T>
  T GetAttrDefaultVal(const std::string&) const;

 private:
  const UserOpDef::ArgDef* GetArgPointer(const std::string&) const;

  UserOpDef def_;
  HashMap<std::string, UserOpDef::ArgDef*> inputs_;
  HashMap<std::string, UserOpDef::ArgDef*> outputs_;
  HashMap<std::string, UserOpDef::AttrDef*> attrs_;
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_USER_OP_DEF_WRAPPER_H_
