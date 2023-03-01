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
#include "oneflow/core/framework/user_op_registry.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/stream.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/attr_value.h"
#include "oneflow/core/framework/attr_value_accessor.h"
#include "oneflow/core/framework/sbp_context.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace user_op {

namespace {

bool InsertIfNotExists(const std::string& name, HashSet<std::string>* unique_names) {
  if (unique_names->find(name) != unique_names->end()) { return false; }
  unique_names->emplace(name);
  return true;
}

}  // namespace

OpRegistry& OpRegistry::Name(const std::string& op_type_name) {
  CHECK(InsertIfNotExists(op_type_name, &unique_names_));
  result_.op_type_name = op_type_name;
  return *this;
}

OpRegistry& OpRegistry::ArgImpl(bool is_input, const std::string& name, bool is_optional) {
  CHECK(InsertIfNotExists(name, &unique_names_))
      << "op arg registered, name: " << name << ", op: " << result_.op_type_name;
  UserOpDef::ArgDef arg_def;
  {
    arg_def.set_name(name);
    arg_def.set_is_optional(is_optional);
  }
  if (is_input) {
    *(result_.op_def.mutable_input()->Add()) = arg_def;
  } else {
    *(result_.op_def.mutable_output()->Add()) = arg_def;
  }
  return *this;
}

#define OP_REG_ARG_MEMBER_FUNC(name_prefix, is_input, is_optional) \
  OpRegistry& OpRegistry::name_prefix(const std::string& name) {   \
    return ArgImpl(is_input, name, is_optional);                   \
  }

OP_REG_ARG_MEMBER_FUNC(Input, true, false)
OP_REG_ARG_MEMBER_FUNC(OptionalInput, true, true)
OP_REG_ARG_MEMBER_FUNC(Output, false, false)
OP_REG_ARG_MEMBER_FUNC(OptionalOutput, false, true)

#undef OP_REG_ARG_MEMBER_FUNC

OpRegistry& OpRegistry::SupportCpuOnly() {
  result_.cpu_only_supported = true;
  return *this;
}

OpRegistry& OpRegistry::SupportNonContiguous() {
  result_.non_contiguous_supported = true;
  return *this;
}

OpRegistry& OpRegistry::NoGrad() {
  result_.no_grad = true;
  return *this;
}

OpRegistry& OpRegistry::SetOutputBufferNum(int32_t num) {
  result_.same_output_regst_num = num;
  return *this;
}

OpRegistry& OpRegistry::Attr(const std::string& name, AttrType type) {
  CHECK(InsertIfNotExists(name, &unique_names_));
  UserOpDef::AttrDef attr_def;
  attr_def.set_name(name);
  attr_def.set_type(type);
  *(result_.op_def.mutable_attr()->Add()) = attr_def;
  return *this;
}

namespace {

void AddAttrWithDefault(OpRegistryResult* result, const std::string& name, AttrType type,
                        std::function<void(UserOpDef::AttrDef*)> handler) {
  UserOpDef::AttrDef attr_def;
  attr_def.set_name(name);
  attr_def.set_type(type);
  handler(&attr_def);
  *(result->op_def.mutable_attr()->Add()) = std::move(attr_def);
}

}  // namespace

#define ATTR_MEMBER_FUNC(field, cpp_type, attr_type)                                             \
  template<>                                                                                     \
  OpRegistry& OpRegistry::Attr<cpp_type>(const std::string& name, AttrType type,                 \
                                         const cpp_type& default_val) {                          \
    CHECK_EQ(type, attr_type);                                                                   \
    return DefaultedAttr(name, type, [default_val](UserOpDef::AttrDef* attr_def) {               \
      AttrValueAccessor<cpp_type>::Attr(default_val, attr_def->mutable_default_val());           \
    });                                                                                          \
  }                                                                                              \
  template<>                                                                                     \
  OpRegistry& OpRegistry::Attr<cpp_type>(const std::string& name, const cpp_type& default_val) { \
    return DefaultedAttr(                                                                        \
        name, GetAttrType<cpp_type>::value, [default_val](UserOpDef::AttrDef* attr_def) {        \
          AttrValueAccessor<cpp_type>::Attr(default_val, attr_def->mutable_default_val());       \
        });                                                                                      \
  }                                                                                              \
  template<>                                                                                     \
  OpRegistry& OpRegistry::Attr<cpp_type>(const std::string& name) {                              \
    return Attr<cpp_type>(name, cpp_type());                                                     \
  }

OF_PP_FOR_EACH_TUPLE(ATTR_MEMBER_FUNC, ATTR_SEQ)

#undef ATTR_MEMBER_FUNC

OpRegistry& OpRegistry::DefaultedAttr(const std::string& name, AttrType type,
                                      const std::function<void(UserOpDef::AttrDef*)>& SetDefault) {
  CHECK(InsertIfNotExists(name, &unique_names_));
  AddAttrWithDefault(&result_, name, type, SetDefault);
  return *this;
}

OpRegistry& OpRegistry::SetTensorDescInferFn(TensorDescInferFn tensor_desc_infer_fn) {
  SetLogicalTensorDescInferFn(tensor_desc_infer_fn);
  SetPhysicalTensorDescInferFn(tensor_desc_infer_fn);
  return *this;
}

OpRegistry& OpRegistry::SetLogicalTensorDescInferFn(TensorDescInferFn tensor_desc_infer_fn) {
  result_.logical_tensor_desc_infer_fn = std::move(tensor_desc_infer_fn);
  return *this;
}

OpRegistry& OpRegistry::SetPhysicalTensorDescInferFn(TensorDescInferFn tensor_desc_infer_fn) {
  result_.physical_tensor_desc_infer_fn = std::move(tensor_desc_infer_fn);
  return *this;
}

OpRegistry& OpRegistry::SetCheckAttrFn(CheckAttrFn fn) {
  result_.check_fn = std::move(fn);
  return *this;
}

OpRegistry& OpRegistry::SetGetSbpFn(GetSbpFn get_sbp_fn) {
  result_.get_sbp_fn = std::move(get_sbp_fn);
  return *this;
}

OpRegistry& OpRegistry::SetSbpSignatureInferFn(SbpSignatureInferFn sbp_signature_infer_fn) {
  result_.sbp_signature_infer_fn = std::move(sbp_signature_infer_fn);
  return *this;
}

OpRegistry& OpRegistry::SetInputArgModifyFn(InputArgModifyFn input_arg_modify_fn) {
  result_.input_arg_modify_fn = std::move(input_arg_modify_fn);
  return *this;
}

OpRegistry& OpRegistry::SetOutputArgModifyFn(OutputArgModifyFn output_arg_modify_fn) {
  result_.output_arg_modify_fn = std::move(output_arg_modify_fn);
  return *this;
}

OpRegistry& OpRegistry::SetOutputBlobTimeShapeInferFn(
    OutputBlobTimeShapeInferFn output_blob_time_shape_infer_fn) {
  result_.output_blob_time_shape_infer_fn = std::move(output_blob_time_shape_infer_fn);
  return *this;
}

OpRegistry& OpRegistry::SetNdSbpInferFn(NdSbpInferFn nd_sbp_infer_fn) {
  result_.nd_sbp_infer_fn = std::move(nd_sbp_infer_fn);
  return *this;
}

OpRegistry& OpRegistry::SetDataTypeInferFn(DataTypeInferFn data_type_infer_fn) {
  result_.data_type_infer_fn = std::move(data_type_infer_fn);
  return *this;
}

OpRegistry& OpRegistry::SetDeviceAndStreamInferFn(
    DeviceAndStreamInferFn device_and_stream_infer_fn) {
  result_.device_and_stream_infer_fn = std::move(device_and_stream_infer_fn);
  return *this;
}

OpRegistry& OpRegistry::SetComputeComplexityFn(ComputeComplexityFn compute_complexity_fn) {
  result_.compute_complexity_fn = std::move(compute_complexity_fn);
  return *this;
}

OpRegistry& OpRegistry::SetGetNdSbpSignatureListFn(GetNdSbpSignatureListFn get_nd_sbp_list_fn) {
  result_.get_nd_sbp_list_fn = std::move(get_nd_sbp_list_fn);
  return *this;
}

OpRegistry& OpRegistry::SetEnumerateNdSbpSignaturesFn(EnumerateNdSbpSignaturesFn fn) {
  result_.enumerate_nd_sbp_signatures_fn = std::move(fn);
  return *this;
}

OpRegistry& OpRegistry::SetDumpNdSbpSignatureForOpConfFn(
    Operator::DumpNdSbpSignatureForOpConfFn fn) {
  result_.dump_nd_sbp_signature_for_op_conf_fn = std::move(fn);
  return *this;
}

Maybe<OpRegistry&> OpRegistry::Finish() {
  CHECK_OR_RETURN(result_.logical_tensor_desc_infer_fn != nullptr)
      << "No TensorDescInfer function for " << result_.op_type_name;
  if (!result_.physical_tensor_desc_infer_fn) {
    const auto& logical_fn = result_.logical_tensor_desc_infer_fn;
    result_.physical_tensor_desc_infer_fn =
        [logical_fn](user_op::InferContext* ctx) -> Maybe<void> {
      if (ctx->parallel_num() == 1) {
        logical_fn(ctx);
      } else {
        for (const auto& pair : ctx->inputs()) {
          const auto& nd_sbp = ctx->NdSbp4ArgNameAndIndex(pair.first, pair.second);
          const TensorDesc* in_logical =
              ctx->LogicalTensorDesc4ArgNameAndIndex(pair.first, pair.second);
          const TensorDesc& in_physical = ctx->InputTensorDesc(pair.first, pair.second);
          CHECK_OR_RETURN(*JUST(GetPhysicalShape(in_logical->shape(), nd_sbp, ctx->parallel_desc(),
                                                 ctx->parallel_ctx()))
                          == in_physical.shape());
        }
        for (const auto& pair : ctx->outputs()) {
          TensorDesc* desc = ctx->MutOutputTensorDesc(pair.first, pair.second);
          *desc = *ctx->LogicalTensorDesc4ArgNameAndIndex(pair.first, pair.second);
          const auto& nd_sbp = ctx->NdSbp4ArgNameAndIndex(pair.first, pair.second);
          desc->set_shape(*JUST(
              GetPhysicalShape(desc->shape(), nd_sbp, ctx->parallel_desc(), ctx->parallel_ctx())));
          desc->set_stride(Stride(desc->shape()));
        }
      }
      return Maybe<void>::Ok();
    };
  }
  if (result_.check_fn == nullptr) { result_.check_fn = CheckAttrFnUtil::NoCheck; }
  CHECK_OR_RETURN(result_.get_sbp_fn != nullptr) << "No Sbp function for " << result_.op_type_name;
  if (result_.cpu_only_supported && result_.device_and_stream_infer_fn == nullptr) {
    result_.device_and_stream_infer_fn =
        [](DeviceAndStreamInferContext* ctx) -> Maybe<Symbol<Stream>> {
      for (const auto& pair : ctx->inputs()) {
        const Symbol<Device>& input_device =
            ctx->InputTensorDevice4ArgNameAndIndex(pair.first, pair.second);
        CHECK_EQ(input_device->type(), "cpu");
      }
      Symbol<Device> default_device;
      {
        if (ctx->inputs().size() != 0) {
          const auto& first_input_name = ctx->inputs().begin()->first;
          default_device = ctx->InputTensorDevice4ArgNameAndIndex(first_input_name, 0);
        } else {
          default_device = JUST(Device::New("cpu"));
        }
      }
      for (const auto& pair : ctx->outputs()) {
        *ctx->OutputTensorDevice4ArgNameAndIndex(pair.first, pair.second) = default_device;
      }
      return Stream::New(default_device, StreamType::kCompute);
    };
  }
  return *this;
}

}  // namespace user_op

}  // namespace oneflow
