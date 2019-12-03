#ifndef ONEFLOW_CORE_FRAMEWORK_OP_REGISTRATION_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_REGISTRATION_H_

#include "oneflow/core/framework/registrar.h"
#include "oneflow/core/framework/user_op_def.pb.h"
#include "oneflow/core/framework/user_op_attr.pb.h"
#include "oneflow/core/framework/user_op_conf.pb.h"
#include "oneflow/core/framework/infer_util.h"

namespace oneflow {

class BlobDesc;
class SbpSignatureList;

namespace user_op {

class UserOpDefWrapper;
class UserOpConfWrapper;

using CheckAttrFn = std::function<Maybe<void>(const UserOpDefWrapper&, const UserOpConfWrapper&)>;
using ShapeInferFn = std::function<Maybe<void>(InferContext*)>;
using DtypeInferFn = std::function<Maybe<void>(InferContext*)>;
using GetSbpFn = std::function<Maybe<void>(
    std::function<Maybe<const BlobDesc*>(const std::string&)>, SbpSignatureList*)>;

struct OpRegistrationVal {
  UserOpDef op_def;
  CheckAttrFn check_fn;
  ShapeInferFn shape_infer_fn;
  DtypeInferFn dtype_infer_fn;
  GetSbpFn get_sbp_fn;
};

struct OpRegistryWrapper final {
  void InsertToGlobalRegistry();

  std::string op_type_name;
  OpRegistrationVal reg_val;
};

class OpRegistryWrapperBuilder final {
 public:
  OpRegistryWrapperBuilder(const std::string& op_type_name);
  OpRegistryWrapperBuilder& Input(const std::string& name);
  OpRegistryWrapperBuilder& Input(const std::string& name, int32_t num);
  OpRegistryWrapperBuilder& InputWithMinimum(const std::string& name, int32_t min_num);
  OpRegistryWrapperBuilder& OptionalInput(const std::string& name);
  OpRegistryWrapperBuilder& OptionalInput(const std::string& name, int32_t num);
  OpRegistryWrapperBuilder& OptionalInputWithMinimum(const std::string& name, int32_t min_num);

  OpRegistryWrapperBuilder& Output(const std::string& name);
  OpRegistryWrapperBuilder& Output(const std::string& name, int32_t num);
  OpRegistryWrapperBuilder& OutputWithMinimum(const std::string& name, int32_t min_num);
  OpRegistryWrapperBuilder& OptionalOutput(const std::string& name);
  OpRegistryWrapperBuilder& OptionalOutput(const std::string& name, int32_t num);
  OpRegistryWrapperBuilder& OptionalOutputWithMinimum(const std::string& name, int32_t min_num);

  OpRegistryWrapperBuilder& Attr(const std::string& name, UserOpAttrType type);
  template<typename T>
  OpRegistryWrapperBuilder& Attr(const std::string& name, UserOpAttrType type, T&& default_val);

  OpRegistryWrapperBuilder& SetShapeInferFn(ShapeInferFn fn);
  OpRegistryWrapperBuilder& SetDataTypeInferFn(DtypeInferFn fn);
  OpRegistryWrapperBuilder& SetGetSbpFn(GetSbpFn fn);
  OpRegistryWrapperBuilder& SetCheckAttrFn(CheckAttrFn fn);

  OpRegistryWrapper Build();

 private:
  OpRegistryWrapperBuilder& ArgImpl(bool is_input, const std::string& name, bool is_optional,
                                    int32_t num, bool num_as_min);

  OpRegistryWrapper wrapper_;
  HashSet<std::string> unique_names_;
};

const OpRegistrationVal* LookUpInOpRegistry(const std::string& op_type_name);

std::vector<std::string> GetAllUserOpInOpRegistry();

}  // namespace user_op

}  // namespace oneflow

#define REGISTER_USER_OP(name)                                                                  \
  static ::oneflow::user_op::Registrar<::oneflow::user_op::OpRegistryWrapperBuilder> OF_PP_CAT( \
      g_registrar, __COUNTER__) = ::oneflow::user_op::OpRegistryWrapperBuilder(name)

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_REGISTRATION_H_
