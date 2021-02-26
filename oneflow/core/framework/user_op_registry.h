/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
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
class InferSbpSignatureFnContext;
class InferOutputBlobTimeShapeFnContext;

using CheckAttrFn = std::function<Maybe<void>(const UserOpDefWrapper&, const UserOpConfWrapper&)>;
using TensorDescInferFn = std::function<Maybe<void>(InferContext*)>;
using GetSbpFn = std::function<Maybe<void>(SbpContext*)>;
using InferSbpSignatureFn = std::function<Maybe<void>(InferSbpSignatureFnContext*)>;
using InputArgModifier = InputBlobModifier;
using GetInputArgModifier =
    std::function<InputArgModifier*(const std::string& in_arg_name, int32_t in_arg_index)>;
using InputArgModifyFn = std::function<void(GetInputArgModifier, const UserOpConfWrapper&)>;
using OutputArgModifier = OutputBlobModifier;
using GetOutputArgModifier =
    std::function<OutputArgModifier*(const std::string& out_arg_name, int32_t out_arg_index)>;
using OutputArgModifyFn = std::function<void(GetOutputArgModifier, const UserOpConfWrapper&)>;
using InferOutputBlobTimeShapeFn = std::function<Maybe<void>(InferOutputBlobTimeShapeFnContext*)>;

struct OpRegistryResult {
  OpRegistryResult() : cpu_only_supported(false), same_output_regst_num(-1) {}
  ~OpRegistryResult() = default;

  std::string op_type_name;
  bool cpu_only_supported;
  int32_t same_output_regst_num;
  UserOpDef op_def;
  CheckAttrFn check_fn;
  TensorDescInferFn tensor_desc_infer_fn;
  GetSbpFn get_sbp_fn;
  InferSbpSignatureFn infer_sbp_signature_fn;
  // TODO(niuchong): move input_arg_modify_fn out of OpRegistryResult since it is more about
  // performance other than op definition
  InputArgModifyFn input_arg_modify_fn;
  OutputArgModifyFn output_arg_modify_fn;
  InferOutputBlobTimeShapeFn infer_output_blob_time_shape_fn;
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

  __attribute__((deprecated)) OpRegistry& Attr(const std::string& name, AttrType type);
  template<typename T>
  __attribute__((deprecated)) OpRegistry& Attr(const std::string& name, AttrType type,
                                               const T& default_val);
  template<typename T>
  OpRegistry& Attr(const std::string& name, const T& default_val);
  template<typename T>
  OpRegistry& Attr(const std::string& name);

  OpRegistry& SetTensorDescInferFn(TensorDescInferFn fn);
  OpRegistry& SetGetSbpFn(GetSbpFn fn);
  OpRegistry& SetInferSbpSignatureFn(InferSbpSignatureFn fn);
  OpRegistry& SetInputArgModifyFn(InputArgModifyFn fn);
  OpRegistry& SetOutputArgModifyFn(OutputArgModifyFn fn);
  OpRegistry& SetInferOutputBlobTimeShapeFn(InferOutputBlobTimeShapeFn fn);
  OpRegistry& SetCheckAttrFn(CheckAttrFn fn);

  OpRegistry& Finish();
  OpRegistryResult GetResult() { return result_; }

 private:
  OpRegistry& ArgImpl(bool is_input, const std::string& name, bool is_optional, int32_t num,
                      bool num_as_min);
  OpRegistry& DefaultedAttr(const std::string& name, AttrType type,
                            const std::function<void(UserOpDef::AttrDef*)>& SetDefault);

 private:
  HashSet<std::string> unique_names_;
  OpRegistryResult result_;
};

static const std::string kUserSourceOpTickInputArgName = "UserSourceOpTickInput";

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_REGISTRY_H_
