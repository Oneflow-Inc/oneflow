#ifndef ONEFLOW_CORE_FRAMEWORK_OP_REGISTRY_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_REGISTRY_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/user_op_def.pb.h"
#include "oneflow/core/framework/user_op_attr.pb.h"
#include "oneflow/core/framework/user_op_conf.pb.h"
#include "oneflow/core/operator/op_attribute.pb.h"

namespace oneflow {

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

struct OpRegistryResult {
  OpRegistryResult() : cpu_only_supported(false), same_output_regst_num(-1) {}
  ~OpRegistryResult() = default;

  std::string op_type_name;
  bool cpu_only_supported;
  int32_t same_output_regst_num;
  UserOpDef op_def;
  CheckAttrFn check_fn;
  TensorDescInferFn tensor_desc_infer_fn;
  BatchAxisInferFn batch_axis_infer_fn;
  GetSbpFn get_sbp_fn;
  // TODO(niuchong): move input_arg_modify_fn out of OpRegistryResult since it is more about
  // performance other than op definition
  InputArgModifyFn input_arg_modify_fn;
  OutputArgModifyFn output_arg_modify_fn;
};

class OpRegistry final {
 public:
  OpRegistry& Name(const std::string& op_type_name);

  OpRegistry& Input(const std::string& name);
  OpRegistry& Input(const std::string& name, int32_t num);
  OpRegistry& InputWithMinimum(const std::string& name, int32_t min_num);
  OpRegistry& OptionalInput(const std::string& name);
  OpRegistry& OptionalInput(const std::string& name, int32_t num);
  OpRegistry& OptionalInputWithMinimum(const std::string& name, int32_t min_num);

  OpRegistry& Output(const std::string& name);
  OpRegistry& Output(const std::string& name, int32_t num);
  OpRegistry& OutputWithMinimum(const std::string& name, int32_t min_num);
  OpRegistry& OptionalOutput(const std::string& name);
  OpRegistry& OptionalOutput(const std::string& name, int32_t num);
  OpRegistry& OptionalOutputWithMinimum(const std::string& name, int32_t min_num);

  OpRegistry& SupportCpuOnly();
  OpRegistry& SetOutputBufferNum(int32_t num);

  OpRegistry& Attr(const std::string& name, UserOpAttrType type);
  template<typename T>
  OpRegistry& Attr(const std::string& name, UserOpAttrType type, T&& default_val);

  OpRegistry& SetTensorDescInferFn(TensorDescInferFn fn);
  OpRegistry& SetBatchAxisInferFn(BatchAxisInferFn fn);
  OpRegistry& SetGetSbpFn(GetSbpFn fn);
  OpRegistry& SetInputArgModifyFn(InputArgModifyFn fn);
  OpRegistry& SetOutputArgModifyFn(OutputArgModifyFn fn);
  OpRegistry& SetCheckAttrFn(CheckAttrFn fn);

  OpRegistry& Finish();
  OpRegistryResult GetResult() { return result_; }

 private:
  OpRegistry& ArgImpl(bool is_input, const std::string& name, bool is_optional, int32_t num,
                      bool num_as_min);

 private:
  HashSet<std::string> unique_names_;
  OpRegistryResult result_;
};

static const std::string kUserSourceOpTickInputArgName = "UserSourceOpTickInput";

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_REGISTRY_H_