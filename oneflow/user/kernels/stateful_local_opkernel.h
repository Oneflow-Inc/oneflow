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
#ifndef ONEFLOW_USER_KERNELS_STATEFUL_LOCAL_OPKERNEL_H_
#define ONEFLOW_USER_KERNELS_STATEFUL_LOCAL_OPKERNEL_H_

#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/tensor_meta.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/framework/stream.h"
#include "oneflow/core/framework/user_op_kernel_registry.h"
#include "oneflow/core/framework/arg_tuple.h"
#include "oneflow/core/framework/op_interpreter.h"

namespace oneflow {

class AttrMap;

namespace vm {
struct LocalCallOpKernelUtil;
}  // namespace vm

namespace one {

class LocalUserKernelBaseContext;
class LocalUserKernelRegContext;
class LocalUserKernelInitAndCacheContext;
class LocalUserOpInferContext;

class ConsistentTensorInferResult;

using ArgVec = std::vector<std::pair<std::string, int32_t>>;

using EagerBlobObjectListRawPtr = const std::vector<std::shared_ptr<vm::EagerBlobObject>>*;
using ConsistentTensorInferResultRawPtr = const ConsistentTensorInferResult*;

class EagerBlobObjectTensorView final : public user_op::Tensor {
 public:
  EagerBlobObjectTensorView(const std::function<vm::EagerBlobObject*()>& mut_eager_blob_object)
      : mut_eager_blob_object_(mut_eager_blob_object) {}

  ShapeView shape() const override { return mut_eager_blob_object_()->shape(); }

  MutShapeView mut_shape() override { return mut_eager_blob_object_()->mut_shape(); }

  const Stride& stride() const override { return mut_eager_blob_object_()->stride(); }

  DataType data_type() const override { return mut_eager_blob_object_()->data_type(); }

  const MemoryCase& mem_case() const override { return mut_eager_blob_object_()->mem_case(); }

  const void* raw_dptr() const override { return mut_eager_blob_object_()->dptr(); }

  void* mut_raw_dptr() override { return mut_eager_blob_object_()->mut_dptr(); }

 private:
  const std::function<vm::EagerBlobObject*()> mut_eager_blob_object_;
};

class EagerBlobObjectTensorDescView final : public user_op::TensorDesc {
 public:
  EagerBlobObjectTensorDescView(const std::function<vm::EagerBlobObject*()>& mut_eager_blob_object)
      : mut_eager_blob_object_(mut_eager_blob_object) {}

  const Shape& shape() const override { return mut_eager_blob_object_()->shape(); }

  Shape* mut_shape() override { return &mut_eager_blob_object_()->mut_shape(); }

  const Stride& stride() const override { return mut_eager_blob_object_()->stride(); }

  Stride* mut_stride() override { return &mut_eager_blob_object_()->mut_stride(); }

  DataType data_type() const override { return mut_eager_blob_object_()->data_type(); }

  DataType* mut_data_type() override { return mut_eager_blob_object_()->mut_data_type(); }

  bool is_dynamic() const override { return mut_eager_blob_object_()->is_dynamic(); }

  bool* mut_is_dynamic() override { return mut_eager_blob_object_()->mut_is_dynamic(); }

  void set_is_dynamic(bool val) override { mut_eager_blob_object_()->set_is_dynamic(val); }

 private:
  const std::function<vm::EagerBlobObject*()> mut_eager_blob_object_;
};

class ConsistentTensorMetaTensorDescView final : public user_op::TensorDesc {
 public:
  ConsistentTensorMetaTensorDescView(
      const std::function<Symbol<ConsistentTensorMeta>()>& consistent_tensor_meta)
      : consistent_tensor_meta_(consistent_tensor_meta) {}

  const Shape& shape() const override { return consistent_tensor_meta_()->shape(); }

  Shape* mut_shape() override {
    UNIMPLEMENTED();
    return nullptr;
  }

  const Stride& stride() const override { return consistent_tensor_meta_()->stride(); }

  Stride* mut_stride() override { UNIMPLEMENTED(); }

  DataType data_type() const override { return consistent_tensor_meta_()->data_type(); }

  DataType* mut_data_type() override {
    UNIMPLEMENTED();
    return nullptr;
  }

  bool is_dynamic() const override { return false; }

  bool* mut_is_dynamic() override {
    UNIMPLEMENTED();
    return nullptr;
  }

  void set_is_dynamic(bool val) override { UNIMPLEMENTED(); }

  Symbol<NdSbp> nd_sbp() { return consistent_tensor_meta_()->nd_sbp(); }

 private:
  const std::function<Symbol<ConsistentTensorMeta>()> consistent_tensor_meta_;
};

class ZeroCopyBaseContext {
 public:
  ZeroCopyBaseContext(const std::shared_ptr<const ArgTuple>& input_arg_tuple,
                      const std::shared_ptr<const ArgTuple>& output_arg_tuple);
  ZeroCopyBaseContext(const std::shared_ptr<const ArgTuple>& input_arg_tuple,
                      const std::shared_ptr<const ArgTuple>& output_arg_tuple,
                      vm::EagerBlobObject* tmp_buffer);

  user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name, int32_t index) const;

  user_op::Tensor* Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t index) const;

  const ConsistentTensorMeta* ConsistentTensorMeta4ArgNameAndIndex(const std::string& arg_name,
                                                                   const int32_t index) const;

  const ConsistentTensorMetaTensorDescView* ConsistentTensorMetaView4ArgNameAndIndex(
      const std::string& arg_name, const int32_t index) const;

  Optional<Symbol<ParallelDesc>> parallel_desc() const;
  const ParallelContext& parallel_ctx() const;

  const ArgVec& inputs() const { return input_arg_tuple_->indexed_arg_name_and_index(); }
  const ArgVec& outputs() const { return output_arg_tuple_->indexed_arg_name_and_index(); }

  void Update(EagerBlobObjectListRawPtr inputs, EagerBlobObjectListRawPtr outputs,
              ConsistentTensorInferResultRawPtr consistent_tensor_infer_result);

 private:
  std::shared_ptr<const ArgTuple> input_arg_tuple_;
  std::shared_ptr<const ArgTuple> output_arg_tuple_;
  std::vector<std::unique_ptr<EagerBlobObjectTensorView>> input_tensor_views_;
  std::vector<std::unique_ptr<EagerBlobObjectTensorView>> output_tensor_views_;
  std::vector<std::unique_ptr<EagerBlobObjectTensorDescView>> input_tensor_desc_views_;
  std::vector<std::unique_ptr<EagerBlobObjectTensorDescView>> output_tensor_desc_views_;
  std::unique_ptr<EagerBlobObjectTensorView> tmp_buffer_view_;
  EagerBlobObjectListRawPtr input_tensors_;
  EagerBlobObjectListRawPtr output_tensors_;
  ConsistentTensorInferResultRawPtr consistent_tensor_infer_result_;
  std::vector<std::unique_ptr<ConsistentTensorMetaTensorDescView>>
      input_consistent_tensor_meta_views_;
  std::vector<std::unique_ptr<ConsistentTensorMetaTensorDescView>>
      output_consistent_tensor_meta_views_;
  ;
};

class LocalUserKernelBaseContext : public ZeroCopyBaseContext {
 public:
  LocalUserKernelBaseContext(const std::string& device_tag,
                             const std::shared_ptr<const ArgTuple>& input_tensor_tuple,
                             const std::shared_ptr<const ArgTuple>& output_tensor_tuple);
  LocalUserKernelBaseContext(const std::string& device_tag,
                             const std::shared_ptr<const ArgTuple>& input_tensor_tuple,
                             const std::shared_ptr<const ArgTuple>& output_tensor_tuple,
                             vm::EagerBlobObject* tmp_buffer);
  ~LocalUserKernelBaseContext() = default;

  DeviceType device_type() const { return device_type_; }
  const std::string& device_tag() const { return device_tag_; }
  const JobDesc& job_desc() const {
    UNIMPLEMENTED();
    return *(const JobDesc*)nullptr;
  }

 private:
  const std::string device_tag_;
  const DeviceType device_type_;
  vm::EagerBlobObject* tmp_buffer_;
};

class LocalUserOpInferContext : public user_op::InferContext {
 public:
  LocalUserOpInferContext(const user_op::UserOpConfWrapper* user_op_conf,
                          const ComposedAttrMap* composed_attrs,
                          const std::shared_ptr<const ArgTuple>& input_arg_tuple,
                          const std::shared_ptr<const ArgTuple>& output_arg_tuple);
  ~LocalUserOpInferContext() override = default;

  const user_op::TensorDesc* LogicalTensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                               int32_t index) const override {
    UNIMPLEMENTED();
    return nullptr;
  }

  const user_op::TensorDesc& InputTensorDesc(const std::string& arg_name,
                                             int32_t index) const override {
    auto out =
        const_cast<LocalUserOpInferContext*>(this)->TensorDesc4ArgNameAndIndex(arg_name, index);
    CHECK_NOTNULL(out);
    return *out;
  }
  user_op::TensorDesc* OutputTensorDesc(const std::string& arg_name, int32_t index) override {
    return TensorDesc4ArgNameAndIndex(arg_name, index);
  }
  user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name, int32_t index);
  const Shape& InputShape(const std::string& arg_name, int32_t index) const override {
    return *const_cast<LocalUserOpInferContext*>(this)->Shape4ArgNameAndIndex(arg_name, index);
  }
  Shape* OutputShape(const std::string& arg_name, int32_t index) override {
    return Shape4ArgNameAndIndex(arg_name, index);
  }
  Shape* Shape4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return NonNullTensorDesc4ArgNameAndIndex(arg_name, index)->mut_shape();
  }
  const Stride& InputStride(const std::string& arg_name, int32_t index) const override {
    return *const_cast<LocalUserOpInferContext*>(this)->Stride4ArgNameAndIndex(arg_name, index);
  }
  Stride* OutputStride(const std::string& arg_name, int32_t index) override {
    return Stride4ArgNameAndIndex(arg_name, index);
  }
  Stride* Stride4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return NonNullTensorDesc4ArgNameAndIndex(arg_name, index)->mut_stride();
  }
  const DataType& InputDType(const std::string& arg_name, int32_t index) const override {
    return *const_cast<LocalUserOpInferContext*>(this)->Dtype4ArgNameAndIndex(arg_name, index);
  }
  DataType* OutputDType(const std::string& arg_name, int32_t index) override {
    return Dtype4ArgNameAndIndex(arg_name, index);
  }
  DataType* Dtype4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return NonNullTensorDesc4ArgNameAndIndex(arg_name, index)->mut_data_type();
  }
  bool InputIsDynamic(const std::string& arg_name, int32_t index) const override {
    return *const_cast<LocalUserOpInferContext*>(this)->IsDynamic4ArgNameAndIndex(arg_name, index);
  }
  bool* OutputIsDynamic(const std::string& arg_name, int32_t index) override {
    return IsDynamic4ArgNameAndIndex(arg_name, index);
  }
  bool* IsDynamic4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return NonNullTensorDesc4ArgNameAndIndex(arg_name, index)->mut_is_dynamic();
  }

  const ArgVec& inputs() const override { return zero_copy_base_ctx_.inputs(); }
  const ArgVec& outputs() const override { return zero_copy_base_ctx_.outputs(); }
  const JobDesc* job_desc() const override {
    UNIMPLEMENTED();
    return nullptr;
  }
  const ParallelContext& parallel_ctx() const override {
    return zero_copy_base_ctx_.parallel_ctx();
  }
  const ParallelDesc& parallel_desc() const override {
    return *CHECK_JUST(zero_copy_base_ctx_.parallel_desc());
  }
  const SbpParallel& SbpParallel4ArgNameAndIndex(const std::string& arg_name,
                                                 int32_t index) const override {
    const auto& nd_sbp = NdSbp4ArgNameAndIndex(arg_name, index);
    CHECK_EQ(nd_sbp.sbp_parallel_size(), 1);
    return nd_sbp.sbp_parallel(0);
  }
  const NdSbp& NdSbp4ArgNameAndIndex(const std::string& arg_name, int32_t index) const override {
    return *CHECK_NOTNULL(zero_copy_base_ctx_.ConsistentTensorMeta4ArgNameAndIndex(arg_name, index))
                ->nd_sbp();
  }

  int64_t parallel_num() const override { return parallel_ctx().parallel_num(); }

  void Update(EagerBlobObjectListRawPtr inputs, EagerBlobObjectListRawPtr outputs,
              ConsistentTensorInferResultRawPtr consistent_tensor_infer_result);

  const std::string& input(const std::string& arg_name, int32_t index) const override {
    return user_op_conf().input(arg_name, index);
  }
  const std::string& output(const std::string& arg_name, int32_t index) const override {
    return user_op_conf().output(arg_name, index);
  }
  bool has_input(const std::string& arg_name, int32_t index) const override {
    return user_op_conf().has_input(arg_name, index);
  }
  bool has_output(const std::string& arg_name, int32_t index) const override {
    return user_op_conf().has_output(arg_name, index);
  }
  int32_t input_size(const std::string& arg_name) const override {
    return user_op_conf().input_size(arg_name);
  }
  int32_t output_size(const std::string& arg_name) const override {
    return user_op_conf().output_size(arg_name);
  }
  const std::string& op_name() const override { return user_op_conf().op_name(); }
  const std::string& op_type_name() const override { return user_op_conf().op_type_name(); }
  const std::string& device_tag() const override { return user_op_conf().op_conf().device_tag(); }
  const std::string& op_loc() const override { return user_op_conf_->op_conf().loc(); }

 private:
  user_op::TensorDesc* NonNullTensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                         int32_t index) {
    user_op::TensorDesc* tensor_desc = TensorDesc4ArgNameAndIndex(arg_name, index);
    if (!tensor_desc) { LOG(FATAL) << "Arg (" << arg_name << "," << index << ") is not found"; }
    return tensor_desc;
  }
  const user_op::UserOpConfWrapper& user_op_conf() const { return *user_op_conf_; }
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return composed_attrs_->Attr4Name(attr_name);
  }

  const user_op::UserOpConfWrapper* user_op_conf_;
  const ComposedAttrMap* composed_attrs_;
  ZeroCopyBaseContext zero_copy_base_ctx_;
};

class LocalUserKernelComputeContext final : public user_op::KernelComputeContext {
 public:
  explicit LocalUserKernelComputeContext(DeviceCtx* device_ctx, const std::string& device_tag,
                                         const user_op::UserOpConfWrapper* user_op_conf,
                                         const ComposedAttrMap* composed_attrs,
                                         const std::shared_ptr<const ArgTuple>& input_arg_tuple,
                                         const std::shared_ptr<const ArgTuple>& output_arg_tuple,
                                         vm::EagerBlobObject* tmp_buffer);
  ~LocalUserKernelComputeContext() = default;

  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return base_ctx_.TensorDesc4ArgNameAndIndex(arg_name, index);
  }

  user_op::Tensor* Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return base_ctx_.Tensor4ArgNameAndIndex(arg_name, index);
  }
  ep::Stream* stream() override {
    CHECK(device_ctx_);
    return device_ctx_->stream();
  }

  DeviceType device_type() const override { return base_ctx_.device_type(); }
  const ParallelContext& parallel_ctx() const override { return base_ctx_.parallel_ctx(); }

  const ArgVec& inputs() const override { return base_ctx_.inputs(); };
  const ArgVec& outputs() const override { return base_ctx_.outputs(); };

  void Update(EagerBlobObjectListRawPtr inputs, EagerBlobObjectListRawPtr outputs,
              ConsistentTensorInferResultRawPtr consistent_tensor_infer_result,
              DeviceCtx* device_ctx);

 private:
  const user_op::UserOpConfWrapper& user_op_conf() const override { return *user_op_conf_; }
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return composed_attrs_->Attr4Name(attr_name);
  }

  const user_op::UserOpConfWrapper* user_op_conf_;
  const ComposedAttrMap* composed_attrs_;
  DeviceCtx* device_ctx_;
  LocalUserKernelBaseContext base_ctx_;
};

class StatefulLocalOpKernel final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StatefulLocalOpKernel);
  static Maybe<StatefulLocalOpKernel> New(const std::shared_ptr<OperatorConf>& op_conf,
                                          const Symbol<Stream>& stream, const AttrMap& base_attrs,
                                          const std::shared_ptr<const ParallelDesc>& parallel_desc,
                                          const std::shared_ptr<const ArgTuple>& input_arg_tuple,
                                          const std::shared_ptr<const ArgTuple>& output_arg_tuple);
  ~StatefulLocalOpKernel();
  const Symbol<Stream>& stream() const { return stream_; }
  const std::shared_ptr<MemoryCase>& mem_case() const { return stream_->device()->mem_case(); }
  const std::string& op_type_name() const { return op_conf_->user_conf().op_type_name(); }
  const std::vector<int64_t>& input_tuple_indexes4const_ibns() const {
    return input_tuple_indexes4const_ibns_;
  }
  const std::vector<int64_t>& input_tuple_indexes4mut_ibns() const {
    return input_tuple_indexes4mut_ibns_;
  }
  const std::vector<int64_t>& output_tuple_indexes4mut_obns() const {
    return output_tuple_indexes4mut_obns_;
  }
  const std::vector<int64_t>& output_tuple_indexes4mut2_obns() const {
    return output_tuple_indexes4mut2_obns_;
  }

  ComposedAttrMap* composed_attrs_for_scheduler_thread() const {
    return composed_attrs_for_scheduler_thread_.get();
  }

  ComposedAttrMap* composed_attrs_for_main_thread() const {
    return composed_attrs_for_main_thread_.get();
  }

  LocalUserOpInferContext* op_infer_ctx_for_scheduler_thread() const {
    return op_infer_ctx_for_scheduler_thread_.get();
  }

  void set_need_check_mem_case(bool value) { need_check_mem_case_ = value; }

  Maybe<void> ChooseOpKernel(const user_op::OpKernel** user_opkernel, bool* need_temp_storage,
                             const AttrMap& attrs, EagerBlobObjectListRawPtr inputs,
                             EagerBlobObjectListRawPtr outputs,
                             ConsistentTensorInferResultRawPtr consistent_tensor_infer_result);

  const OperatorConf& op_conf() const { return *op_conf_; }

 private:
  friend struct vm::LocalCallOpKernelUtil;
  StatefulLocalOpKernel() = default;
  LocalUserKernelComputeContext* UpdateComputeContext(
      EagerBlobObjectListRawPtr inputs, EagerBlobObjectListRawPtr outputs,
      ConsistentTensorInferResultRawPtr consistent_tensor_infer_result, DeviceCtx* device_ctx);

  user_op::TensorDescInferFn TensorDescInferFn() const;
  user_op::DataTypeInferFn DataTypeInferFn() const;

  void TryInitOpKernelStateAndCache(
      const user_op::OpKernel* op_kernel, DeviceCtx* device_ctx, EagerBlobObjectListRawPtr inputs,
      EagerBlobObjectListRawPtr outputs,
      ConsistentTensorInferResultRawPtr consistent_tensor_infer_result,
      user_op::OpKernelState** state, user_op::OpKernelCache** cache);

  vm::EagerBlobObject* mut_temp_blob_object();

  user_op::OpKernelState* mut_opkernel_state(const user_op::OpKernel* opkernel) {
    return op_kernel_state_map_.at(opkernel).get();
  }

  bool need_check_mem_case() const { return need_check_mem_case_; }

  const user_op::InferTmpSizeFn& GetInferTmpSizeFn(const user_op::OpKernel* op_kernel) const;

  std::shared_ptr<OperatorConf> op_conf_;
  std::unique_ptr<ComposedAttrMap> composed_attrs_for_scheduler_thread_;
  std::unique_ptr<ComposedAttrMap> composed_attrs_for_main_thread_;
  std::unique_ptr<user_op::UserOpConfWrapper> user_op_conf_;
  Symbol<Stream> stream_;
  std::unique_ptr<LocalUserKernelRegContext> reg_ctx_;
  std::unique_ptr<LocalUserOpInferContext> op_infer_ctx_for_scheduler_thread_;
  std::unique_ptr<LocalUserKernelComputeContext> compute_ctx_;
  std::shared_ptr<const ArgTuple> input_arg_tuple_;
  std::shared_ptr<const ArgTuple> output_arg_tuple_;
  bool need_check_mem_case_;
  user_op::TensorDescInferFn tensor_desc_infer_fn_;
  user_op::DataTypeInferFn data_type_infer_fn_;
  // NOTE: every device has its own stateful local opkernel instance,
  // so only group kernels by dtype
  std::array<std::vector<std::pair<const user_op::OpKernelRegistryResult*,
                                   std::shared_ptr<const user_op::OpKernel>>>,
             DataType_MAX>
      dtype2cached_kernels_;
  HashMap<const user_op::OpKernel*, std::shared_ptr<user_op::OpKernelState>> op_kernel_state_map_;
  HashMap<const user_op::OpKernel*, std::shared_ptr<user_op::OpKernelCache>> op_kernel_cache_map_;
  HashMap<const user_op::OpKernel*, const user_op::InferTmpSizeFn*> infer_tmp_size_fn_map_;
  std::unique_ptr<vm::EagerBlobObject> tmp_blob_object_;
  std::vector<int64_t> input_tuple_indexes4const_ibns_;
  std::vector<int64_t> input_tuple_indexes4mut_ibns_;
  std::vector<int64_t> output_tuple_indexes4mut_obns_;
  std::vector<int64_t> output_tuple_indexes4mut2_obns_;
};

}  // namespace one

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_STATEFUL_LOCAL_OPKERNEL_H_
