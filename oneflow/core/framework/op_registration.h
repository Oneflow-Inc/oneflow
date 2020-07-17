#ifndef ONEFLOW_CORE_FRAMEWORK_OP_REGISTRATION_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_REGISTRATION_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/registrar.h"
#include "oneflow/core/framework/user_op_def.pb.h"
#include "oneflow/core/framework/user_op_attr.pb.h"
#include "oneflow/core/framework/user_op_conf.pb.h"
#include "oneflow/core/operator/op_attribute.pb.h"

namespace oneflow {

class TensorDesc;
class SbpSignatureList;

namespace user_op {

class UserOpDefWrapper;
class UserOpConfWrapper;
class InferContext;
class SbpContext;
class BatchAxisContext;

using CheckAttrFn = std::function<Maybe<void>(const UserOpDefWrapper&, const UserOpConfWrapper&)>;
using TensorDescInferFn = std::function<Maybe<void>(InferContext*)>;
using BatchAxisInferFn = std::function<Maybe<void>(BatchAxisContext*)>;
using GetSbpFn = std::function<Maybe<void>(SbpContext*)>;
using InputArgModifier = InputBlobModifier;
using GetInputArgModifier =
    std::function<InputArgModifier*(const std::string& in_arg_name, int32_t in_arg_index)>;
using InputArgModifyFn = std::function<void(GetInputArgModifier, const UserOpConfWrapper&)>;
using OutputArgModifier = OutputBlobModifier;
using GetOutputArgModifier =
    std::function<OutputArgModifier*(const std::string& out_arg_name, int32_t out_arg_index)>;
using OutputArgModifyFn = std::function<void(GetOutputArgModifier, const UserOpConfWrapper&)>;

struct OpRegistrationVal {
  OpRegistrationVal() : cpu_only_supported(false), same_output_regst_num(-1) {}
  ~OpRegistrationVal() = default;
  bool cpu_only_supported;
  int32_t same_output_regst_num;
  UserOpDef op_def;
  CheckAttrFn check_fn;
  TensorDescInferFn tensor_desc_infer_fn;
  BatchAxisInferFn batch_axis_infer_fn;
  GetSbpFn get_sbp_fn;
  // TODO(niuchong): move input_arg_modify_fn out of OpRegistrationVal since it is more about
  // performance other than op definition
  InputArgModifyFn input_arg_modify_fn;
  OutputArgModifyFn output_arg_modify_fn;
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

  OpRegistryWrapperBuilder& SupportCpuOnly();
  OpRegistryWrapperBuilder& SetOutputBufferNum(int32_t num);

  OpRegistryWrapperBuilder& Attr(const std::string& name, UserOpAttrType type);
  template<typename T>
  OpRegistryWrapperBuilder& Attr(const std::string& name, UserOpAttrType type, T&& default_val);

  OpRegistryWrapperBuilder& SetTensorDescInferFn(TensorDescInferFn fn);
  OpRegistryWrapperBuilder& SetBatchAxisInferFn(BatchAxisInferFn fn);
  OpRegistryWrapperBuilder& SetGetSbpFn(GetSbpFn fn);
  OpRegistryWrapperBuilder& SetInputArgModifyFn(InputArgModifyFn fn);
  OpRegistryWrapperBuilder& SetOutputArgModifyFn(OutputArgModifyFn fn);
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

static const std::string kUserSourceOpTickInputArgName = "UserSourceOpTickInput";

}  // namespace user_op

}  // namespace oneflow

#define REGISTER_USER_OP(name)                                                                  \
  static ::oneflow::user_op::Registrar<::oneflow::user_op::OpRegistryWrapperBuilder> OF_PP_CAT( \
      g_registrar, __COUNTER__) = ::oneflow::user_op::OpRegistryWrapperBuilder(name)

#define REGISTER_CPU_ONLY_USER_OP(name) REGISTER_USER_OP(name).SupportCpuOnly()

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_REGISTRATION_H_
