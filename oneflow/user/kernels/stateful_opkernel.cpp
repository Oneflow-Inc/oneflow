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
#include "oneflow/user/kernels/stateful_opkernel.h"
#include "oneflow/core/framework/attr_value_accessor.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/framework/global_tensor_infer_cache.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/profiler/profile_manager.h"
#include "oneflow/core/profiler/event_recorder.h"
#include "oneflow/core/eager/call_context.h"

namespace oneflow {
namespace one {

class GlobalTensorInferResult;

using ArgVec = std::vector<std::pair<std::string, int32_t>>;

using EagerBlobObjectListRawPtr = const std::vector<std::shared_ptr<vm::EagerBlobObject>>*;
using GlobalTensorInferResultRawPtr = const GlobalTensorInferResult*;

class ZeroCopyBaseContextHelper {
 public:
  ZeroCopyBaseContextHelper(const std::shared_ptr<const ArgTuple>& input_arg_tuple,
                            const std::shared_ptr<const ArgTuple>& output_arg_tuple)
      : input_arg_tuple_(input_arg_tuple), output_arg_tuple_(output_arg_tuple) {}

#define RETURN_IF_FOUND(inputs, outputs, post_action)                                             \
  int32_t i = TryGetTensorTupleIndex(input_arg_tuple_->arg_name2bn_index2tensor_tuple_index(),    \
                                     arg_name, index);                                            \
  if (i >= 0) { return (inputs).at(i) post_action; }                                              \
  i = TryGetTensorTupleIndex(output_arg_tuple_->arg_name2bn_index2tensor_tuple_index(), arg_name, \
                             index);                                                              \
  if (i >= 0) { return (outputs).at(i) post_action; }

  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(eager::CallContext* call_ctx,
                                                        const std::string& arg_name,
                                                        const int32_t index) const {
    RETURN_IF_FOUND(call_ctx->inputs(), call_ctx->outputs(), .get());
    return nullptr;
  }
  user_op::TensorDesc* MutTensorDesc4ArgNameAndIndex(eager::CallContext* call_ctx,
                                                     const std::string& arg_name,
                                                     const int32_t index) const {
    RETURN_IF_FOUND(call_ctx->inputs(), call_ctx->outputs(), .get());
    return nullptr;
  }

  user_op::Tensor* Tensor4ArgNameAndIndex(eager::CallContext* call_ctx, const std::string& arg_name,
                                          const int32_t index) const {
    RETURN_IF_FOUND(call_ctx->inputs(), call_ctx->outputs(), .get());
    if (arg_name == "tmp_buffer" && index == 0) { return call_ctx->mut_tmp_tensor(); }
    return nullptr;
  }

  const GlobalTensorMeta* GlobalTensorMeta4ArgNameAndIndex(eager::CallContext* call_ctx,
                                                           const std::string& arg_name,
                                                           const int32_t index) const {
    const auto& global_tensor_infer_result = call_ctx->global_tensor_infer_result();
    RETURN_IF_FOUND(global_tensor_infer_result->input_tensor_metas(),
                    global_tensor_infer_result->output_tensor_metas(), .shared_from_symbol().get());
    return nullptr;
  }

  Optional<Symbol<ParallelDesc>> parallel_desc(eager::CallContext* call_ctx) const {
    const auto& global_tensor_infer_result = call_ctx->global_tensor_infer_result();
    if (!global_tensor_infer_result) { return Optional<Symbol<ParallelDesc>>(); }
    if (!global_tensor_infer_result->input_tensor_metas().empty()) {
      return global_tensor_infer_result->input_tensor_metas().at(0)->parallel_desc();
    } else if (!global_tensor_infer_result->output_tensor_metas().empty()) {
      return global_tensor_infer_result->output_tensor_metas().at(0)->parallel_desc();
    } else {
      UNIMPLEMENTED();
      return Optional<Symbol<ParallelDesc>>();
    }
  }

  const ParallelContext& parallel_ctx(eager::CallContext* call_ctx) const {
    const auto& parallel_desc = this->parallel_desc(call_ctx);
    if (parallel_desc.has_value()) {
      const auto& parallel_desc_symbol = CHECK_JUST(parallel_desc);
      return *CHECK_JUST(GetParallelContext4CurrentProcessCtx(parallel_desc_symbol));
    } else {
      static ParallelContext single_device_parallel_ctx(MakeSingleDeviceParallelCtx());
      return single_device_parallel_ctx;
    }
  }

  const ArgVec& inputs() const { return input_arg_tuple_->indexed_arg_name_and_index(); }
  const ArgVec& outputs() const { return output_arg_tuple_->indexed_arg_name_and_index(); }

 private:
  static int32_t TryGetTensorTupleIndex(const std::unordered_map<std::string, std::vector<int32_t>>&
                                            arg_name2bn_index2tensor_tuple_index,
                                        const std::string& arg_name, const int32_t arg_index) {
    auto it = arg_name2bn_index2tensor_tuple_index.find(arg_name);
    if (it != arg_name2bn_index2tensor_tuple_index.end()) { return it->second.at(arg_index); }
    return -1;
  }

  static ParallelContext MakeSingleDeviceParallelCtx() {
    ParallelContext single_device_parallel_ctx;
    single_device_parallel_ctx.set_parallel_id(0);
    single_device_parallel_ctx.set_parallel_num(1);
    return single_device_parallel_ctx;
  }

  std::shared_ptr<const ArgTuple> input_arg_tuple_;
  std::shared_ptr<const ArgTuple> output_arg_tuple_;
};

class UserKernelBaseContextHelper final : public ZeroCopyBaseContextHelper {
 public:
  UserKernelBaseContextHelper(DeviceType device_type,
                              const std::shared_ptr<const ArgTuple>& input_arg_tuple,
                              const std::shared_ptr<const ArgTuple>& output_arg_tuple)
      : ZeroCopyBaseContextHelper(input_arg_tuple, output_arg_tuple), device_type_(device_type) {}

  ~UserKernelBaseContextHelper() = default;

  DeviceType device_type() const { return device_type_; }
  const JobDesc& job_desc() const {
    UNIMPLEMENTED();
    return *(const JobDesc*)nullptr;
  }

 private:
  const DeviceType device_type_;
};

class UserOpInferContextHelper final {
 public:
  UserOpInferContextHelper(const user_op::UserOpConfWrapper* user_op_conf,
                           const std::shared_ptr<const ArgTuple>& input_arg_tuple,
                           const std::shared_ptr<const ArgTuple>& output_arg_tuple)
      : user_op_conf_(user_op_conf),
        zero_copy_base_ctx_helper_(input_arg_tuple, output_arg_tuple) {}

  ~UserOpInferContextHelper() = default;

  const user_op::TensorDesc* LogicalTensorDesc4ArgNameAndIndex(eager::CallContext* call_ctx,
                                                               const std::string& arg_name,
                                                               int32_t index) const {
    UNIMPLEMENTED();
    return nullptr;
  }

  const user_op::TensorDesc& InputTensorDesc(eager::CallContext* call_ctx,
                                             const std::string& arg_name, int32_t index) const {
    return *TensorDesc4ArgNameAndIndex(call_ctx, arg_name, index);
  }
  const user_op::TensorDesc& OutputTensorDesc(eager::CallContext* call_ctx,
                                              const std::string& arg_name, int32_t index) const {
    return *TensorDesc4ArgNameAndIndex(call_ctx, arg_name, index);
  }
  user_op::TensorDesc* MutOutputTensorDesc(eager::CallContext* call_ctx,
                                           const std::string& arg_name, int32_t index) const {
    return MutTensorDesc4ArgNameAndIndex(call_ctx, arg_name, index);
  }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(eager::CallContext* call_ctx,
                                                        const std::string& arg_name,
                                                        int32_t index) const {
    return zero_copy_base_ctx_helper_.TensorDesc4ArgNameAndIndex(call_ctx, arg_name, index);
  }
  user_op::TensorDesc* MutTensorDesc4ArgNameAndIndex(eager::CallContext* call_ctx,
                                                     const std::string& arg_name,
                                                     int32_t index) const {
    return zero_copy_base_ctx_helper_.MutTensorDesc4ArgNameAndIndex(call_ctx, arg_name, index);
  }

  const Shape& InputShape(eager::CallContext* call_ctx, const std::string& arg_name,
                          int32_t index) const {
    return Shape4ArgNameAndIndex(call_ctx, arg_name, index);
  }
  const Shape& OutputShape(eager::CallContext* call_ctx, const std::string& arg_name,
                           int32_t index) const {
    return Shape4ArgNameAndIndex(call_ctx, arg_name, index);
  }
  void SetOutputShape(eager::CallContext* call_ctx, const std::string& arg_name, int32_t index,
                      const Shape& shape) const {
    SetShape4ArgNameAndIndex(call_ctx, arg_name, index, shape);
  }
  const Shape& Shape4ArgNameAndIndex(eager::CallContext* call_ctx, const std::string& arg_name,
                                     int32_t index) const {
    return NonNullTensorDesc4ArgNameAndIndex(call_ctx, arg_name, index).shape();
  }
  void SetShape4ArgNameAndIndex(eager::CallContext* call_ctx, const std::string& arg_name,
                                int32_t index, const Shape& shape) const {
    return MutNonNullTensorDesc4ArgNameAndIndex(call_ctx, arg_name, index)->set_shape(shape);
  }
  const Stride& InputStride(eager::CallContext* call_ctx, const std::string& arg_name,
                            int32_t index) const {
    return Stride4ArgNameAndIndex(call_ctx, arg_name, index);
  }
  const Stride& OutputStride(eager::CallContext* call_ctx, const std::string& arg_name,
                             int32_t index) const {
    return Stride4ArgNameAndIndex(call_ctx, arg_name, index);
  }
  void SetOutputStride(eager::CallContext* call_ctx, const std::string& arg_name, int32_t index,
                       const Stride& stride) const {
    return SetStride4ArgNameAndIndex(call_ctx, arg_name, index, stride);
  }
  const Stride& Stride4ArgNameAndIndex(eager::CallContext* call_ctx, const std::string& arg_name,
                                       int32_t index) const {
    return NonNullTensorDesc4ArgNameAndIndex(call_ctx, arg_name, index).stride();
  }
  void SetStride4ArgNameAndIndex(eager::CallContext* call_ctx, const std::string& arg_name,
                                 int32_t index, const Stride& stride) const {
    return MutNonNullTensorDesc4ArgNameAndIndex(call_ctx, arg_name, index)->set_stride(stride);
  }
  DataType InputDType(eager::CallContext* call_ctx, const std::string& arg_name,
                      int32_t index) const {
    return Dtype4ArgNameAndIndex(call_ctx, arg_name, index);
  }
  DataType OutputDType(eager::CallContext* call_ctx, const std::string& arg_name,
                       int32_t index) const {
    return Dtype4ArgNameAndIndex(call_ctx, arg_name, index);
  }
  void SetOutputDType(eager::CallContext* call_ctx, const std::string& arg_name, int32_t index,
                      DataType data_type) const {
    return SetDtype4ArgNameAndIndex(call_ctx, arg_name, index, data_type);
  }
  DataType Dtype4ArgNameAndIndex(eager::CallContext* call_ctx, const std::string& arg_name,
                                 int32_t index) const {
    return NonNullTensorDesc4ArgNameAndIndex(call_ctx, arg_name, index).data_type();
  }
  void SetDtype4ArgNameAndIndex(eager::CallContext* call_ctx, const std::string& arg_name,
                                int32_t index, DataType data_type) const {
    return MutNonNullTensorDesc4ArgNameAndIndex(call_ctx, arg_name, index)
        ->set_data_type(data_type);
  }
  bool InputIsDynamic(eager::CallContext* call_ctx, const std::string& arg_name,
                      int32_t index) const {
    return IsDynamic4ArgNameAndIndex(call_ctx, arg_name, index);
  }
  bool OutputIsDynamic(eager::CallContext* call_ctx, const std::string& arg_name,
                       int32_t index) const {
    return IsDynamic4ArgNameAndIndex(call_ctx, arg_name, index);
  }
  void SetOutputIsDynamic(eager::CallContext* call_ctx, const std::string& arg_name, int32_t index,
                          bool is_dynamic) const {
    return SetIsDynamic4ArgNameAndIndex(call_ctx, arg_name, index, is_dynamic);
  }
  bool IsDynamic4ArgNameAndIndex(eager::CallContext* call_ctx, const std::string& arg_name,
                                 int32_t index) const {
    return NonNullTensorDesc4ArgNameAndIndex(call_ctx, arg_name, index).is_dynamic();
  }
  void SetIsDynamic4ArgNameAndIndex(eager::CallContext* call_ctx, const std::string& arg_name,
                                    int32_t index, bool is_dynamic) const {
    return MutNonNullTensorDesc4ArgNameAndIndex(call_ctx, arg_name, index)
        ->set_is_dynamic(is_dynamic);
  }

  const ArgVec& inputs() const { return zero_copy_base_ctx_helper_.inputs(); }
  const ArgVec& outputs() const { return zero_copy_base_ctx_helper_.outputs(); }
  const JobDesc* job_desc() const {
    UNIMPLEMENTED();
    return nullptr;
  }
  const ParallelContext& parallel_ctx(eager::CallContext* call_ctx) const {
    return zero_copy_base_ctx_helper_.parallel_ctx(call_ctx);
  }
  const ParallelDesc& parallel_desc(eager::CallContext* call_ctx) const {
    return *CHECK_JUST(zero_copy_base_ctx_helper_.parallel_desc(call_ctx));
  }
  const SbpParallel& SbpParallel4ArgNameAndIndex(eager::CallContext* call_ctx,
                                                 const std::string& arg_name, int32_t index) const {
    const auto& nd_sbp = NdSbp4ArgNameAndIndex(call_ctx, arg_name, index);
    CHECK_EQ(nd_sbp.sbp_parallel_size(), 1);
    return nd_sbp.sbp_parallel(0);
  }
  const NdSbp& NdSbp4ArgNameAndIndex(eager::CallContext* call_ctx, const std::string& arg_name,
                                     int32_t index) const {
    return *CHECK_NOTNULL(zero_copy_base_ctx_helper_.GlobalTensorMeta4ArgNameAndIndex(
                              call_ctx, arg_name, index))
                ->nd_sbp();
  }

  int64_t parallel_num(eager::CallContext* call_ctx) const {
    return parallel_ctx(call_ctx).parallel_num();
  }

  const std::string& input(const std::string& arg_name, int32_t index) const {
    return user_op_conf().input(arg_name, index);
  }
  const std::string& output(const std::string& arg_name, int32_t index) const {
    return user_op_conf().output(arg_name, index);
  }
  bool has_input(const std::string& arg_name, int32_t index) const {
    return user_op_conf().has_input(arg_name, index);
  }
  bool has_output(const std::string& arg_name, int32_t index) const {
    return user_op_conf().has_output(arg_name, index);
  }
  int32_t input_size(const std::string& arg_name) const {
    return user_op_conf().input_size(arg_name);
  }
  int32_t output_size(const std::string& arg_name) const {
    return user_op_conf().output_size(arg_name);
  }
  const std::string& op_name() const { return user_op_conf().op_name(); }
  const std::string& op_type_name() const { return user_op_conf().op_type_name(); }
  const std::string& op_loc() const { return user_op_conf_->op_conf().loc(); }

  const user_op::UserOpConfWrapper& user_op_conf() const { return *user_op_conf_; }
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(eager::CallContext* call_ctx,
                                                           const std::string& attr_name) const {
    return call_ctx->composed_attrs().Attr4Name(attr_name);
  }

 private:
  const user_op::TensorDesc& NonNullTensorDesc4ArgNameAndIndex(eager::CallContext* call_ctx,
                                                               const std::string& arg_name,
                                                               int32_t index) const {
    const user_op::TensorDesc* tensor_desc = TensorDesc4ArgNameAndIndex(call_ctx, arg_name, index);
    if (!tensor_desc) { LOG(FATAL) << "Arg (" << arg_name << "," << index << ") is not found"; }
    return *tensor_desc;
  }
  user_op::TensorDesc* MutNonNullTensorDesc4ArgNameAndIndex(eager::CallContext* call_ctx,
                                                            const std::string& arg_name,
                                                            int32_t index) const {
    user_op::TensorDesc* tensor_desc = MutTensorDesc4ArgNameAndIndex(call_ctx, arg_name, index);
    if (!tensor_desc) { LOG(FATAL) << "Arg (" << arg_name << "," << index << ") is not found"; }
    return tensor_desc;
  }

  const user_op::UserOpConfWrapper* user_op_conf_;
  ZeroCopyBaseContextHelper zero_copy_base_ctx_helper_;
};

class UserOpInferContext : public user_op::InferContext {
 public:
  UserOpInferContext(const UserOpInferContextHelper* helper, eager::CallContext* call_ctx)
      : helper_(helper), call_ctx_(call_ctx) {}

  ~UserOpInferContext() override = default;

  const user_op::TensorDesc* LogicalTensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                               int32_t index) const override {
    return helper_->LogicalTensorDesc4ArgNameAndIndex(call_ctx_, arg_name, index);
  }

  const user_op::TensorDesc& InputTensorDesc(const std::string& arg_name,
                                             int32_t index) const override {
    return helper_->InputTensorDesc(call_ctx_, arg_name, index);
  }
  const user_op::TensorDesc& OutputTensorDesc(const std::string& arg_name,
                                              int32_t index) const override {
    return helper_->OutputTensorDesc(call_ctx_, arg_name, index);
  }
  user_op::TensorDesc* MutOutputTensorDesc(const std::string& arg_name, int32_t index) override {
    return helper_->MutOutputTensorDesc(call_ctx_, arg_name, index);
  }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const {
    return helper_->TensorDesc4ArgNameAndIndex(call_ctx_, arg_name, index);
  }
  user_op::TensorDesc* MutTensorDesc4ArgNameAndIndex(const std::string& arg_name, int32_t index) {
    return helper_->MutTensorDesc4ArgNameAndIndex(call_ctx_, arg_name, index);
  }

  const Shape& InputShape(const std::string& arg_name, int32_t index) const override {
    return helper_->InputShape(call_ctx_, arg_name, index);
  }
  const Shape& OutputShape(const std::string& arg_name, int32_t index) const override {
    return helper_->OutputShape(call_ctx_, arg_name, index);
  }
  void SetOutputShape(const std::string& arg_name, int32_t index, const Shape& shape) override {
    return helper_->SetOutputShape(call_ctx_, arg_name, index, shape);
  }
  const Shape& Shape4ArgNameAndIndex(const std::string& arg_name, int32_t index) const override {
    return helper_->Shape4ArgNameAndIndex(call_ctx_, arg_name, index);
  }
  void SetShape4ArgNameAndIndex(const std::string& arg_name, int32_t index,
                                const Shape& shape) override {
    return helper_->SetShape4ArgNameAndIndex(call_ctx_, arg_name, index, shape);
  }
  const Stride& InputStride(const std::string& arg_name, int32_t index) const override {
    return helper_->InputStride(call_ctx_, arg_name, index);
  }
  const Stride& OutputStride(const std::string& arg_name, int32_t index) const override {
    return helper_->InputStride(call_ctx_, arg_name, index);
  }
  void SetOutputStride(const std::string& arg_name, int32_t index, const Stride& stride) override {
    return helper_->SetOutputStride(call_ctx_, arg_name, index, stride);
  }
  const Stride& Stride4ArgNameAndIndex(const std::string& arg_name, int32_t index) const override {
    return helper_->Stride4ArgNameAndIndex(call_ctx_, arg_name, index);
  }
  void SetStride4ArgNameAndIndex(const std::string& arg_name, int32_t index,
                                 const Stride& stride) override {
    return helper_->SetStride4ArgNameAndIndex(call_ctx_, arg_name, index, stride);
  }
  DataType InputDType(const std::string& arg_name, int32_t index) const override {
    return helper_->InputDType(call_ctx_, arg_name, index);
  }
  DataType OutputDType(const std::string& arg_name, int32_t index) const override {
    return helper_->OutputDType(call_ctx_, arg_name, index);
  }
  void SetOutputDType(const std::string& arg_name, int32_t index, DataType data_type) override {
    return helper_->SetOutputDType(call_ctx_, arg_name, index, data_type);
  }
  DataType Dtype4ArgNameAndIndex(const std::string& arg_name, int32_t index) const override {
    return helper_->Dtype4ArgNameAndIndex(call_ctx_, arg_name, index);
  }
  void SetDtype4ArgNameAndIndex(const std::string& arg_name, int32_t index,
                                DataType data_type) override {
    return helper_->SetDtype4ArgNameAndIndex(call_ctx_, arg_name, index, data_type);
  }
  bool InputIsDynamic(const std::string& arg_name, int32_t index) const override {
    return helper_->InputIsDynamic(call_ctx_, arg_name, index);
  }
  bool OutputIsDynamic(const std::string& arg_name, int32_t index) const override {
    return helper_->OutputIsDynamic(call_ctx_, arg_name, index);
  }
  void SetOutputIsDynamic(const std::string& arg_name, int32_t index, bool is_dynamic) override {
    return helper_->SetOutputIsDynamic(call_ctx_, arg_name, index, is_dynamic);
  }
  bool IsDynamic4ArgNameAndIndex(const std::string& arg_name, int32_t index) const override {
    return helper_->IsDynamic4ArgNameAndIndex(call_ctx_, arg_name, index);
  }
  void SetIsDynamic4ArgNameAndIndex(const std::string& arg_name, int32_t index,
                                    bool is_dynamic) override {
    return helper_->SetIsDynamic4ArgNameAndIndex(call_ctx_, arg_name, index, is_dynamic);
  }

  const ArgVec& inputs() const override { return helper_->inputs(); }
  const ArgVec& outputs() const override { return helper_->outputs(); }
  const JobDesc* job_desc() const override { return helper_->job_desc(); }
  const ParallelContext& parallel_ctx() const override { return helper_->parallel_ctx(call_ctx_); }
  const ParallelDesc& parallel_desc() const override { return helper_->parallel_desc(call_ctx_); }
  const SbpParallel& SbpParallel4ArgNameAndIndex(const std::string& arg_name,
                                                 int32_t index) const override {
    return helper_->SbpParallel4ArgNameAndIndex(call_ctx_, arg_name, index);
  }
  const NdSbp& NdSbp4ArgNameAndIndex(const std::string& arg_name, int32_t index) const override {
    return helper_->NdSbp4ArgNameAndIndex(call_ctx_, arg_name, index);
  }

  int64_t parallel_num() const override { return helper_->parallel_num(call_ctx_); }

  const std::string& input(const std::string& arg_name, int32_t index) const override {
    return helper_->input(arg_name, index);
  }
  const std::string& output(const std::string& arg_name, int32_t index) const override {
    return helper_->output(arg_name, index);
  }
  bool has_input(const std::string& arg_name, int32_t index) const override {
    return helper_->has_input(arg_name, index);
  }
  bool has_output(const std::string& arg_name, int32_t index) const override {
    return helper_->has_output(arg_name, index);
  }
  int32_t input_size(const std::string& arg_name) const override {
    return helper_->input_size(arg_name);
  }
  int32_t output_size(const std::string& arg_name) const override {
    return helper_->output_size(arg_name);
  }
  const std::string& op_name() const override { return helper_->op_name(); }
  const std::string& op_type_name() const override { return helper_->op_type_name(); }
  const std::string& op_loc() const override { return helper_->op_loc(); }

 private:
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return helper_->Attr4Name(call_ctx_, attr_name);
  }

  const UserOpInferContextHelper* helper_;
  eager::CallContext* call_ctx_;
};

class UserKernelComputeContextHelper final {
 public:
  UserKernelComputeContextHelper(DeviceType device_type,
                                 const user_op::UserOpConfWrapper* user_op_conf,
                                 const std::shared_ptr<const ArgTuple>& input_arg_tuple,
                                 const std::shared_ptr<const ArgTuple>& output_arg_tuple)
      : user_op_conf_(user_op_conf),
        base_ctx_helper_(device_type, input_arg_tuple, output_arg_tuple) {}

  ~UserKernelComputeContextHelper() = default;

  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(eager::CallContext* call_ctx,
                                                        const std::string& arg_name,
                                                        int32_t index) const {
    return base_ctx_helper_.TensorDesc4ArgNameAndIndex(call_ctx, arg_name, index);
  }

  user_op::Tensor* Tensor4ArgNameAndIndex(eager::CallContext* call_ctx, const std::string& arg_name,
                                          int32_t index) const {
    return base_ctx_helper_.Tensor4ArgNameAndIndex(call_ctx, arg_name, index);
  }

  DeviceType device_type() const { return base_ctx_helper_.device_type(); }
  const ParallelContext& parallel_ctx(eager::CallContext* call_ctx) const {
    return base_ctx_helper_.parallel_ctx(call_ctx);
  }

  const ArgVec& inputs() const { return base_ctx_helper_.inputs(); }
  const ArgVec& outputs() const { return base_ctx_helper_.outputs(); }

  const user_op::UserOpConfWrapper& user_op_conf() const { return *user_op_conf_; }
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(eager::CallContext* call_ctx,
                                                           const std::string& attr_name) const {
    return call_ctx->composed_attrs().Attr4Name(attr_name);
  }

 private:
  const user_op::UserOpConfWrapper* user_op_conf_;
  UserKernelBaseContextHelper base_ctx_helper_;
};

class UserKernelComputeContext final : public user_op::KernelComputeContext {
 public:
  UserKernelComputeContext(const UserKernelComputeContextHelper* helper,
                           eager::CallContext* call_ctx, ep::Stream* stream)
      : helper_(helper), call_ctx_(call_ctx), stream_(stream) {}

  ~UserKernelComputeContext() = default;

  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return helper_->TensorDesc4ArgNameAndIndex(call_ctx_, arg_name, index);
  }

  user_op::Tensor* Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return helper_->Tensor4ArgNameAndIndex(call_ctx_, arg_name, index);
  }

  ep::Stream* stream() override {
    CHECK_NOTNULL(stream_);
    return stream_;
  }

  DeviceType device_type() const override { return helper_->device_type(); }

  const ParallelContext& parallel_ctx() const override { return helper_->parallel_ctx(call_ctx_); }

  const ArgVec& inputs() const override { return helper_->inputs(); }
  const ArgVec& outputs() const override { return helper_->outputs(); }

 private:
  const user_op::UserOpConfWrapper& user_op_conf() const override {
    return helper_->user_op_conf();
  }

  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return helper_->Attr4Name(call_ctx_, attr_name);
  }

  const UserKernelComputeContextHelper* helper_;
  eager::CallContext* call_ctx_;
  ep::Stream* stream_;
};

class UserKernelRegContextHelper final {
 public:
  UserKernelRegContextHelper(DeviceType device_type, const user_op::UserOpConfWrapper* user_op_conf,
                             const std::shared_ptr<const ArgTuple>& input_arg_tuple,
                             const std::shared_ptr<const ArgTuple>& output_arg_tuple)
      : user_op_conf_(user_op_conf),
        base_ctx_helper_(device_type, input_arg_tuple, output_arg_tuple) {}
  ~UserKernelRegContextHelper() = default;

  DeviceType device_type() const { return base_ctx_helper_.device_type(); }
  const ParallelContext& parallel_ctx(eager::CallContext* call_ctx) const {
    return base_ctx_helper_.parallel_ctx(call_ctx);
  }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(eager::CallContext* call_ctx,
                                                        const std::string& arg_name,
                                                        int32_t index) const {
    return base_ctx_helper_.TensorDesc4ArgNameAndIndex(call_ctx, arg_name, index);
  }
  const ArgVec& inputs() const { return base_ctx_helper_.inputs(); }
  const ArgVec& outputs() const { return base_ctx_helper_.outputs(); }

  const user_op::UserOpConfWrapper& user_op_conf() const { return *user_op_conf_; }

  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(eager::CallContext* call_ctx,
                                                           const std::string& attr_name) const {
    return call_ctx->composed_attrs().Attr4Name(attr_name);
  }

 private:
  const user_op::UserOpConfWrapper* user_op_conf_;
  UserKernelBaseContextHelper base_ctx_helper_;
};

class UserKernelRegContext final : public user_op::KernelRegContext {
 public:
  UserKernelRegContext(const UserKernelRegContextHelper* helper, eager::CallContext* call_ctx)
      : helper_(helper), call_ctx_(call_ctx) {}
  ~UserKernelRegContext() = default;

  DeviceType device_type() const override { return helper_->device_type(); }
  const ParallelContext& parallel_ctx() const override { return helper_->parallel_ctx(call_ctx_); }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return helper_->TensorDesc4ArgNameAndIndex(call_ctx_, arg_name, index);
  }
  const ArgVec& inputs() const override { return helper_->inputs(); }
  const ArgVec& outputs() const override { return helper_->outputs(); }

  const user_op::UserOpConfWrapper& user_op_conf() const override {
    return helper_->user_op_conf();
  }

 private:
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return helper_->Attr4Name(call_ctx_, attr_name);
  }

  const UserKernelRegContextHelper* helper_;
  eager::CallContext* call_ctx_;
};

class UserKernelInitAndCacheContextHelper final {
 public:
  UserKernelInitAndCacheContextHelper(DeviceType device_type,
                                      const user_op::UserOpConfWrapper* user_op_conf,
                                      const std::shared_ptr<const ArgTuple>& input_arg_tuple,
                                      const std::shared_ptr<const ArgTuple>& output_arg_tuple)
      : user_op_conf_(user_op_conf),
        base_ctx_helper_(device_type, input_arg_tuple, output_arg_tuple) {}

  ~UserKernelInitAndCacheContextHelper() = default;

  DeviceType device_type() const { return base_ctx_helper_.device_type(); }
  const ParallelContext& parallel_ctx(eager::CallContext* call_ctx) const {
    return base_ctx_helper_.parallel_ctx(call_ctx);
  }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(eager::CallContext* call_ctx,
                                                        const std::string& arg_name,
                                                        int32_t index) const {
    return base_ctx_helper_.TensorDesc4ArgNameAndIndex(call_ctx, arg_name, index);
  }
  const user_op::TensorDesc* LogicalTensorDesc4ArgNameAndIndex(eager::CallContext* call_ctx,
                                                               const std::string& arg_name,
                                                               int32_t index) const {
    return base_ctx_helper_.GlobalTensorMeta4ArgNameAndIndex(call_ctx, arg_name, index);
  }
  const SbpParallel& SbpParallel4ArgNameAndIndex(eager::CallContext* call_ctx,
                                                 const std::string& arg_name, int32_t index) const {
    const auto& nd_sbp = NdSbp4ArgNameAndIndex(call_ctx, arg_name, index);
    CHECK_EQ(nd_sbp.sbp_parallel_size(), 1);
    return nd_sbp.sbp_parallel(0);
  }

  const NdSbp& NdSbp4ArgNameAndIndex(eager::CallContext* call_ctx, const std::string& arg_name,
                                     int32_t index) const {
    return *CHECK_NOTNULL(
                base_ctx_helper_.GlobalTensorMeta4ArgNameAndIndex(call_ctx, arg_name, index))
                ->nd_sbp();
  }

  const ArgVec& inputs() const { return base_ctx_helper_.inputs(); }
  const ArgVec& outputs() const { return base_ctx_helper_.outputs(); }
  const ParallelDesc& parallel_desc(eager::CallContext* call_ctx) const {
    return *CHECK_JUST(base_ctx_helper_.parallel_desc(call_ctx));
  }

  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(eager::CallContext* call_ctx,
                                                           const std::string& attr_name) const {
    return call_ctx->composed_attrs().Attr4Name(attr_name);
  }

  const user_op::UserOpConfWrapper& user_op_conf() const { return *user_op_conf_; }

 private:
  const user_op::UserOpConfWrapper* user_op_conf_;
  UserKernelBaseContextHelper base_ctx_helper_;
};

class UserKernelInitAndCacheContext final : public user_op::KernelInitContext,
                                            public user_op::KernelCacheContext {
 public:
  UserKernelInitAndCacheContext(const UserKernelInitAndCacheContextHelper* helper,
                                eager::CallContext* call_ctx, ep::Stream* stream)
      : helper_(helper), call_ctx_(call_ctx), stream_(stream) {}

  ~UserKernelInitAndCacheContext() override = default;

  ep::Stream* stream() override {
    CHECK_NOTNULL(stream_);
    return stream_;
  }

  DeviceType device_type() const override { return helper_->device_type(); }
  const ParallelContext& parallel_ctx() const override { return helper_->parallel_ctx(call_ctx_); }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return helper_->TensorDesc4ArgNameAndIndex(call_ctx_, arg_name, index);
  }
  const user_op::TensorDesc* LogicalTensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                               int32_t index) const override {
    return helper_->LogicalTensorDesc4ArgNameAndIndex(call_ctx_, arg_name, index);
  }
  const SbpParallel& SbpParallel4ArgNameAndIndex(const std::string& arg_name,
                                                 int32_t index) const override {
    return helper_->SbpParallel4ArgNameAndIndex(call_ctx_, arg_name, index);
  }

  const NdSbp& NdSbp4ArgNameAndIndex(const std::string& arg_name, int32_t index) const override {
    return helper_->NdSbp4ArgNameAndIndex(call_ctx_, arg_name, index);
  }

  const ArgVec& inputs() const override { return helper_->inputs(); }
  const ArgVec& outputs() const override { return helper_->outputs(); }
  const ParallelDesc& parallel_desc() const override { return helper_->parallel_desc(call_ctx_); }

 private:
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return helper_->Attr4Name(call_ctx_, attr_name);
  }

  const user_op::UserOpConfWrapper& user_op_conf() const override {
    return helper_->user_op_conf();
  }

  const UserKernelInitAndCacheContextHelper* helper_;
  eager::CallContext* call_ctx_;
  ep::Stream* stream_;
};

namespace {

Maybe<void> InitTensorTupleIndexes4Bns(const std::shared_ptr<const OperatorConf>& op_conf,
                                       const ArgVec& indexed_input_pairs,
                                       const ArgVec& indexed_output_pairs,
                                       OpArgsVector<int64_t>* input_tuple_indexes4const_ibns,
                                       OpArgsVector<int64_t>* input_tuple_indexes4mut_ibns,
                                       OpArgsVector<int64_t>* output_tuple_indexes4mut_obns,
                                       OpArgsVector<int64_t>* output_tuple_indexes4mut2_obns,
                                       small_vector<bool>* output_tuple_indexes2is_mut2_type) {
  const auto* op_reg_val =
      user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_conf->user_conf().op_type_name());
  CHECK_NOTNULL_OR_RETURN(op_reg_val);

  ArgModifierSignature arg_modifier_signature;
  for (const auto& pair : indexed_input_pairs) {
    const std::string ibn = GenRepeatedBn(pair.first, pair.second);
    arg_modifier_signature.mutable_ibn2input_blob_modifier()->insert(
        {ibn, user_op::InputArgModifier()});
  }
  for (const auto& pair : indexed_output_pairs) {
    const std::string obn = GenRepeatedBn(pair.first, pair.second);
    arg_modifier_signature.mutable_obn2output_blob_modifier()->insert(
        {obn, user_op::OutputArgModifier()});
  }
  user_op::UserOpConfWrapper op_conf_wrapper(op_conf);
  if (op_reg_val->input_arg_modify_fn) {
    user_op::GetInputArgModifier GetInputArgModifierFn =
        [&arg_modifier_signature](const std::string& in_arg_name,
                                  int32_t in_arg_index) -> user_op::InputArgModifier* {
      const std::string ibn = GenRepeatedBn(in_arg_name, in_arg_index);
      auto* map = arg_modifier_signature.mutable_ibn2input_blob_modifier();
      return &map->at(ibn);
    };
    JUST(op_reg_val->input_arg_modify_fn(GetInputArgModifierFn, op_conf_wrapper));
  }
  if (op_reg_val->output_arg_modify_fn) {
    user_op::GetOutputArgModifier GetOutputArgModifierFn =
        [&arg_modifier_signature](const std::string& in_arg_name,
                                  int32_t in_arg_index) -> user_op::OutputArgModifier* {
      const std::string obn = GenRepeatedBn(in_arg_name, in_arg_index);
      auto* map = arg_modifier_signature.mutable_obn2output_blob_modifier();
      return &map->at(obn);
    };
    JUST(op_reg_val->output_arg_modify_fn(GetOutputArgModifierFn, op_conf_wrapper));
  }

  for (int i = 0; i < indexed_input_pairs.size(); i++) {
    const auto& pair = indexed_input_pairs.at(i);
    const std::string ibn = GenRepeatedBn(pair.first, pair.second);
    if (arg_modifier_signature.ibn2input_blob_modifier().at(ibn).is_mutable()) {
      input_tuple_indexes4mut_ibns->emplace_back(i);
    } else {
      input_tuple_indexes4const_ibns->emplace_back(i);
    }
  }

  for (int i = 0; i < indexed_output_pairs.size(); i++) {
    const auto& pair = indexed_output_pairs.at(i);
    const std::string obn = GenRepeatedBn(pair.first, pair.second);
    if (arg_modifier_signature.obn2output_blob_modifier().at(obn).header_infered_before_compute()) {
      output_tuple_indexes4mut_obns->emplace_back(i);
      output_tuple_indexes2is_mut2_type->emplace_back(false);
    } else {
      output_tuple_indexes4mut2_obns->emplace_back(i);
      output_tuple_indexes2is_mut2_type->emplace_back(true);
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<StatefulOpKernel> StatefulOpKernel::New(
    const std::shared_ptr<OperatorConf>& op_conf, const Symbol<Stream>& stream,
    const AttrMap& base_attrs, const std::shared_ptr<const ParallelDesc>& parallel_desc,
    const std::shared_ptr<const ArgTuple>& input_arg_tuple,
    const std::shared_ptr<const ArgTuple>& output_arg_tuple) {
  auto opkernel = std::shared_ptr<StatefulOpKernel>(new StatefulOpKernel());
  opkernel->base_attrs_ = base_attrs;
  opkernel->op_conf_ = op_conf;
  opkernel->user_op_conf_.reset(new user_op::UserOpConfWrapper(op_conf));
  opkernel->stream_ = stream;
  opkernel->input_arg_tuple_ = input_arg_tuple;
  opkernel->output_arg_tuple_ = output_arg_tuple;

  const DeviceType device_type = CHECK_JUST(DeviceType4DeviceTag(op_conf->device_tag()));
  const user_op::UserOpConfWrapper* user_op_conf = opkernel->user_op_conf_.get();
  opkernel->op_infer_ctx_helper_.reset(
      new UserOpInferContextHelper(user_op_conf, input_arg_tuple, output_arg_tuple));

  opkernel->init_and_cache_ctx_helper_.reset(new UserKernelInitAndCacheContextHelper(
      device_type, opkernel->user_op_conf_.get(), opkernel->input_arg_tuple_,
      opkernel->output_arg_tuple_));
  opkernel->compute_ctx_helper_.reset(new UserKernelComputeContextHelper(
      device_type, user_op_conf, input_arg_tuple, output_arg_tuple));
  opkernel->reg_ctx_helper_.reset(
      new UserKernelRegContextHelper(device_type, user_op_conf, input_arg_tuple, output_arg_tuple));
  const auto* op_reg_val =
      user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(user_op_conf->op_type_name());
  CHECK_NOTNULL_OR_RETURN(op_reg_val);
  if (op_reg_val->logical_tensor_desc_infer_fn) {
    opkernel->tensor_desc_infer_fn_ = op_reg_val->logical_tensor_desc_infer_fn;
  } else {
    return Error::UnimplementedError();
  }
  opkernel->data_type_infer_fn_ = op_reg_val->data_type_infer_fn;

  JUST(InitTensorTupleIndexes4Bns(
      op_conf, input_arg_tuple->indexed_arg_name_and_index(),
      output_arg_tuple->indexed_arg_name_and_index(), &opkernel->input_tuple_indexes4const_ibns_,
      &opkernel->input_tuple_indexes4mut_ibns_, &opkernel->output_tuple_indexes4mut_obns_,
      &opkernel->output_tuple_indexes4mut2_obns_, &opkernel->output_tuple_indexes2is_mut2_type_));

  return opkernel;
}

StatefulOpKernel::~StatefulOpKernel() = default;

size_t StatefulOpKernel::InferTmpSize(eager::CallContext* call_ctx,
                                      const user_op::OpKernel* user_opkernel) const {
  UserOpInferContext op_infer_ctx(op_infer_ctx_helper_.get(), call_ctx);
  const auto& InferTmpSizeFn = GetInferTmpSizeFn(user_opkernel);
  return InferTmpSizeFn(&op_infer_ctx);
}

Maybe<void> StatefulOpKernel::ChooseOpKernel(eager::CallContext* call_ctx,
                                             const user_op::OpKernel** user_opkernel,
                                             bool* need_temp_storage) {
  DataType primary_dtype = kInvalidDataType;
  const auto& inputs = call_ctx->inputs();
  const auto& outputs = call_ctx->outputs();
  if (likely(!inputs.empty())) {
    primary_dtype = inputs[0]->data_type();
  } else if (likely(!outputs.empty())) {
    primary_dtype = outputs[0]->data_type();
  } else {
    // do nothing
  }

  UserKernelRegContext reg_ctx(reg_ctx_helper_.get(), call_ctx);
  for (const auto& pair : dtype2cached_kernels_[primary_dtype]) {
    if (likely(pair.first->is_matched_hob->get(reg_ctx))) {
      *need_temp_storage = pair.first->need_temp_storage;
      *user_opkernel = pair.second.get();
      return Maybe<void>::Ok();
    }
  }

  OF_PROFILER_RANGE_GUARD("fallback");

  const auto& op_type_name = user_op_conf_->op_type_name();
  const auto* kernel_reg_val =
      JUST(user_op::UserOpRegistryMgr::Get().GetOpKernelRegistryResult(op_type_name, reg_ctx));
  CHECK_NOTNULL(kernel_reg_val);
  auto* kernel = kernel_reg_val->create_fn();
  dtype2cached_kernels_[primary_dtype].push_back(
      {kernel_reg_val, std::shared_ptr<const user_op::OpKernel>(kernel)});

  infer_tmp_size_fn_map_.emplace(kernel, &kernel_reg_val->infer_tmp_size_fn);
  *need_temp_storage = kernel_reg_val->need_temp_storage;
  *user_opkernel = kernel;
  return Maybe<void>::Ok();
}

void StatefulOpKernel::TryInitOpKernelStateAndCache(eager::CallContext* call_ctx,
                                                    ep::Stream* stream,
                                                    const user_op::OpKernel* op_kernel,
                                                    user_op::OpKernelState** state,
                                                    user_op::OpKernelCache** cache) {
  UserKernelInitAndCacheContext init_and_cache_ctx(init_and_cache_ctx_helper_.get(), call_ctx,
                                                   stream);
  if (state != nullptr) {
    auto it = op_kernel_state_map_.find(op_kernel);
    if (it != op_kernel_state_map_.end()) {
      *state = it->second.get();
    } else {
      auto created_state = op_kernel->CreateOpKernelState(&init_and_cache_ctx);
      op_kernel_state_map_.emplace(op_kernel, created_state);
      *state = created_state.get();
    }
  }

  {
    auto& cache_in_map = op_kernel_cache_map_[op_kernel];
    op_kernel->InitOpKernelCacheWithFlags(&init_and_cache_ctx,
                                          user_op::OpKernelCache::kAllMayChanged, &cache_in_map);
    *cache = cache_in_map.get();
  }
}

const user_op::InferTmpSizeFn& StatefulOpKernel::GetInferTmpSizeFn(
    const user_op::OpKernel* op_kernel) const {
  return *infer_tmp_size_fn_map_.at(op_kernel);
}

user_op::TensorDescInferFn StatefulOpKernel::TensorDescInferFn() const {
  return tensor_desc_infer_fn_;
}

user_op::DataTypeInferFn StatefulOpKernel::DataTypeInferFn() const { return data_type_infer_fn_; }

void StatefulOpKernel::Compute(eager::CallContext* call_ctx, ep::Stream* stream,
                               const user_op::OpKernel* user_opkernel,
                               user_op::OpKernelState* state,
                               const user_op::OpKernelCache* cache) const {
  UserKernelComputeContext compute_context(compute_ctx_helper_.get(), call_ctx, stream);
  auto* compute_ctx = &compute_context;
  OF_PROFILER_RANGE_GUARD("Compute");
  auto er_guard = CHECK_JUST(profiler::EventRecorder::CreateKernelEventRecorder(
      op_type_name(),
#if defined(WITH_CUDA)
      [compute_ctx]() -> int64_t {
        const auto CalMemorySize = [compute_ctx](const one::ArgVec& args) -> int64_t {
          const auto Func = [compute_ctx](int64_t mem_size, const auto& pair) {
            const auto tensor = compute_ctx->Tensor4ArgNameAndIndex(pair.first, pair.second);
            return mem_size
                   + tensor->shape_view().elem_cnt() * GetSizeOfDataType(tensor->data_type());
          };
          return std::accumulate(args.begin(), args.end(), static_cast<int64_t>(0), Func);
        };
        return CalMemorySize(compute_ctx->inputs()) + CalMemorySize(compute_ctx->outputs());
      },
#endif
      [call_ctx]() -> std::pair<std::string, int64_t> {
        std::stringstream ss;
        std::size_t hash = 0;
        for (size_t i = 0; i < call_ctx->inputs().size(); i++) {
          const auto& shape = call_ctx->inputs().at(i)->shape();
          ss << shape;
          if (i != call_ctx->inputs().size() - 1) { ss << ", "; }
          AddHash(&hash, shape);
        }
        return {ss.str(), hash};
      },
      [call_ctx]() -> std::pair<std::string, int64_t> {
        const std::string attr_str = call_ctx->composed_attrs().ToString();
        return {attr_str, std::hash<std::string>{}(attr_str)};
      }));
  user_opkernel->Compute(compute_ctx, state, cache);
  CHECK_JUST(compute_ctx->stream()->GetAsyncError());
}

}  // namespace one
}  // namespace oneflow
