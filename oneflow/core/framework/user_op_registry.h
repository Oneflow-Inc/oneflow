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
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/user_op_def.pb.h"
#include "oneflow/core/framework/user_op_attr.pb.h"
#include "oneflow/core/framework/user_op_conf.pb.h"
#include "oneflow/core/operator/op_attribute.pb.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class Device;
class Stream;

namespace user_op {

class UserOpDefWrapper;
class UserOpConfWrapper;
class InferContext;
class SbpContext;
class InferSbpSignatureFnContext;
class InferOutputBlobTimeShapeFnContext;
class InferNdSbpFnContext;
class DeviceAndStreamInferContext;
class ComputeComplexityFnContext;
class GetNdSbpSignatureListContext;

using CheckAttrFn = std::function<Maybe<void>(const UserOpDefWrapper&, const UserOpConfWrapper&)>;
using TensorDescInferFn = std::function<Maybe<void>(InferContext*)>;
using DataTypeInferFn = std::function<Maybe<void>(InferContext*)>;
using DeviceAndStreamInferFn = std::function<Maybe<Symbol<Stream>>(DeviceAndStreamInferContext*)>;
using GetSbpFn = std::function<Maybe<void>(SbpContext*)>;
using SbpSignatureInferFn = std::function<Maybe<void>(InferSbpSignatureFnContext*)>;
using InputArgModifier = InputBlobModifier;
using GetInputArgModifier =
    std::function<InputArgModifier*(const std::string& in_arg_name, int32_t in_arg_index)>;
using InputArgModifyFn =
    std::function<Maybe<void>(const GetInputArgModifier&, const UserOpConfWrapper&)>;
using OutputArgModifier = OutputBlobModifier;
using GetOutputArgModifier =
    std::function<OutputArgModifier*(const std::string& out_arg_name, int32_t out_arg_index)>;
using OutputArgModifyFn =
    std::function<Maybe<void>(const GetOutputArgModifier&, const UserOpConfWrapper&)>;
using OutputBlobTimeShapeInferFn = std::function<Maybe<void>(InferOutputBlobTimeShapeFnContext*)>;
using NdSbpInferFn = std::function<Maybe<void>(InferNdSbpFnContext*)>;
using ComputeComplexityFn = std::function<Maybe<double>(ComputeComplexityFnContext*)>;
// TODO: set up another context
using GetNdSbpSignatureListFn = std::function<Maybe<void>(GetNdSbpSignatureListContext*)>;
using EnumerateNdSbpSignaturesFn = std::function<Maybe<void>(GetNdSbpSignatureListContext*)>;

struct OpRegistryResult {
  OpRegistryResult()
      : cpu_only_supported(false),
        no_grad(false),
        non_contiguous_supported(false),
        same_output_regst_num(-1) {}
  ~OpRegistryResult() = default;

  std::string op_type_name;
  bool cpu_only_supported;
  bool no_grad;
  bool non_contiguous_supported;
  int32_t same_output_regst_num;
  UserOpDef op_def;
  CheckAttrFn check_fn;
  TensorDescInferFn logical_tensor_desc_infer_fn;
  TensorDescInferFn physical_tensor_desc_infer_fn;
  GetSbpFn get_sbp_fn;
  SbpSignatureInferFn sbp_signature_infer_fn;
  DataTypeInferFn data_type_infer_fn;
  DeviceAndStreamInferFn device_and_stream_infer_fn;
  // TODO(niuchong): move input_arg_modify_fn out of OpRegistryResult since it is more about
  // performance other than op definition
  InputArgModifyFn input_arg_modify_fn;
  OutputArgModifyFn output_arg_modify_fn;
  OutputBlobTimeShapeInferFn output_blob_time_shape_infer_fn;
  NdSbpInferFn nd_sbp_infer_fn;
  ComputeComplexityFn compute_complexity_fn;
  GetNdSbpSignatureListFn get_nd_sbp_list_fn;
  EnumerateNdSbpSignaturesFn enumerate_nd_sbp_signatures_fn;
  Operator::DumpNdSbpSignatureForOpConfFn dump_nd_sbp_signature_for_op_conf_fn;
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
  OpRegistry& SupportNonContiguous();
  OpRegistry& NoGrad();
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
  OpRegistry& SetLogicalTensorDescInferFn(TensorDescInferFn fn);
  OpRegistry& SetPhysicalTensorDescInferFn(TensorDescInferFn fn);
  OpRegistry& SetGetSbpFn(GetSbpFn fn);
  OpRegistry& SetSbpSignatureInferFn(SbpSignatureInferFn fn);
  OpRegistry& SetInputArgModifyFn(InputArgModifyFn fn);
  OpRegistry& SetOutputArgModifyFn(OutputArgModifyFn fn);
  OpRegistry& SetOutputBlobTimeShapeInferFn(OutputBlobTimeShapeInferFn fn);
  OpRegistry& SetNdSbpInferFn(NdSbpInferFn fn);
  OpRegistry& SetCheckAttrFn(CheckAttrFn fn);
  OpRegistry& SetDataTypeInferFn(DataTypeInferFn fn);
  OpRegistry& SetDeviceAndStreamInferFn(DeviceAndStreamInferFn fn);
  OpRegistry& SetComputeComplexityFn(ComputeComplexityFn fn);
  OpRegistry& SetGetNdSbpSignatureListFn(GetNdSbpSignatureListFn fn);
  OpRegistry& SetEnumerateNdSbpSignaturesFn(EnumerateNdSbpSignaturesFn fn);
  OpRegistry& SetDumpNdSbpSignatureForOpConfFn(Operator::DumpNdSbpSignatureForOpConfFn fn);

  Maybe<OpRegistry&> Finish();
  OpRegistryResult GetResult() { return result_; }

 private:
  OpRegistry& ArgImpl(bool is_input, const std::string& name, bool is_optional);
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
