#ifndef ONEFLOW_CORE_FRAMEWORK_OP_REGISTRATION_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_REGISTRATION_H_

#include "oneflow/core/framework/registrar.h"
#include "oneflow/core/framework/user_op_def.pb.h"
#include "oneflow/core/framework/user_op_attr.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

namespace user_op {

using Shape4ArgNameAndIndex = std::function<Shape*(const std::string&, int32_t)>;
using Dtype4ArgNameAndIndex = std::function<DataType*(const std::string&, int32_t)>;

struct OpRegistrationVal {
  UserOpDef op_def;
  std::function<Maybe<void>(const UserOpDef&, const UserOpConf&)> check_fn;
  std::function<Maybe<void>(Shape4ArgNameAndIndex)> shape_infer_fn;
  std::function<Maybe<void>(Dtype4ArgNameAndIndex)> dtype_infer_fn;
  std::function<Maybe<void>(/*TODO(niuchong): what is the para*/)> get_sbp_fn;
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
  OpRegistryWrapperBuilder& Attr(const std::string& name, UserOpAttrType type, T&& default_val) {
    // TODO(niuchong): move this to cpp and specialization this func
    CHECK_EQ(type, GetAttrType<T>::value);
    UserOpDef::AttrDef attr_def;
    attr_def.name = name;
    attr_def.type = type;
    switch (type) {
      case UserOpAttrType::kAtInt32:
        attr_def.mutable_default_val()->set_at_int32(default_val);
        break;
      case UserOpAttrType::kAtInt64:
        attr_def.mutable_default_val()->set_at_int64(default_val);
        break;
      case UserOpAttrType::kAtBool: attr_def.mutable_default_val()->set_at_bool(default_val); break;
      case UserOpAttrType::kAtFloat:
        attr_def.mutable_default_val()->set_at_float(default_val);
        break;
      case UserOpAttrType::kAtDouble:
        attr_def.mutable_default_val()->set_at_double(default_val);
        break;
      case UserOpAttrType::kAtString:
        attr_def.mutable_default_val()->set_at_string(default_val);
        break;
      case UserOpAttrType::kAtShape:
        default_val.ToProto(attr_def.mutable_default_val()->mutable_at_shape());
        break;
      case UserOpAttrType::kAtListInt32:
        SerializeVector2ListAttr<std::vector<int32_t>, UserOpAttrVal::ListInt32>(
            default_val, attr_def.mutable_default_val());
        break;
      case UserOpAttrType::kAtListInt64:
        SerializeVector2ListAttr<std::vector<int64_t>, UserOpAttrVal::ListInt64>(
            default_val, attr_def.mutable_default_val());
        break;
      case UserOpAttrType::kAtListFloat:
        SerializeVector2ListAttr<std::vector<float>, UserOpAttrVal::ListFloat>(
            default_val, attr_def.mutable_default_val());
        break;
      default: UNIMPLEMENTED();
    }

    *(wrapper_.reg_val.op_def.mutable_attr()->Add()) = attr_def;
    return *this;
  }

  OpRegistryWrapperBuilder& SetShapeInferFn(std::function<Maybe<void>(Shape4ArgNameAndIndex)>);
  OpRegistryWrapperBuilder& SetDataTypeInferFn(std::function<Maybe<void>(Dtype4ArgNameAndIndex)>);
  OpRegistryWrapperBuilder& SetGetSbpFn(
      std::function<Maybe<void>(/*TODO(niuchong): what is the para*/)>) {}

  OpRegistryWrapper Build() { return wrapper_; }

 private:
  OpRegistryWrapperBuilder& ArgImpl(bool is_input, const std::string& name, bool is_optional,
                                    int32_t num, bool num_as_min);

  OpRegistryWrapper wrapper_;
};

}  // namespace user_op

#define REGISTER_USER_OP(name)                                                              \
  static user_op::Registrar<OpRegistryWrapperBuilder> OF_PP_CAT(g_registrar, __COUNTER__) = \
      user_op::OpRegistryWrapperBuilder(name)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_REGISTRATION_H_
