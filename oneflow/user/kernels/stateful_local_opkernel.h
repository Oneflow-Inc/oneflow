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
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/framework/user_op_kernel_registry.h"
#include "oneflow/core/framework/arg_tuple.h"

namespace oneflow {

class AttrMap;

namespace vm {
struct LocalCallOpKernelUtil;
}  // namespace vm

namespace one {

class LocalUserKernelBaseContext;
class LocalUserKernelRegContext;
class LocalUserKernelCreateContext;
class LocalUserKernelInitContext;
class LocalUserOpInferContext;

using ArgVec = std::vector<std::pair<std::string, int32_t>>;

using EagerBlobObjectListPtr =
    std::shared_ptr<const std::vector<std::shared_ptr<vm::EagerBlobObject>>>;

class EagerBlobObjectTensorView final : public user_op::Tensor {
 public:
  EagerBlobObjectTensorView(const std::function<vm::EagerBlobObject*()>& mut_eager_blob_object)
      : mut_eager_blob_object_(mut_eager_blob_object) {}

  const ShapeView& shape() const override { return mut_eager_blob_object_()->blob().shape(); }

  MutShapeView* mut_shape() override {
    return mut_eager_blob_object_()->mut_blob()->mut_shape_view();
  }

  DataType data_type() const override { return mut_eager_blob_object_()->blob().data_type(); }

  const MemoryCase& mem_case() const override {
    return mut_eager_blob_object_()->blob().mem_case();
  }

  const void* raw_dptr() const override { return mut_eager_blob_object_()->blob().dptr(); }

  void* mut_raw_dptr() override { return mut_eager_blob_object_()->mut_blob()->mut_dptr(); }

 private:
  const std::function<vm::EagerBlobObject*()> mut_eager_blob_object_;
};

class EagerBlobObjectTensorDescView final : public user_op::TensorDesc {
 public:
  EagerBlobObjectTensorDescView(const std::function<vm::EagerBlobObject*()>& mut_eager_blob_object)
      : mut_eager_blob_object_(mut_eager_blob_object) {}

  const Shape& shape() const override { return mut_eager_blob_object_()->blob_desc().shape(); }

  Shape* mut_shape() override { return &mut_eager_blob_object_()->mut_blob_desc()->mut_shape(); }

  DataType data_type() const override { return mut_eager_blob_object_()->blob_desc().data_type(); }

  DataType* mut_data_type() override {
    return mut_eager_blob_object_()->mut_blob_desc()->mut_data_type();
  }

  bool is_dynamic() const override { return mut_eager_blob_object_()->blob_desc().is_dynamic(); }

  bool* mut_is_dynamic() override {
    return mut_eager_blob_object_()->mut_blob_desc()->mut_is_dynamic();
  }

  void set_is_dynamic(bool val) override {
    mut_eager_blob_object_()->mut_blob_desc()->set_is_dynamic(val);
  }

 private:
  const std::function<vm::EagerBlobObject*()> mut_eager_blob_object_;
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

  const ArgVec& inputs() const { return input_arg_tuple_->indexed_arg_name_and_index(); }
  const ArgVec& outputs() const { return output_arg_tuple_->indexed_arg_name_and_index(); }

  void Update(const EagerBlobObjectListPtr& inputs, const EagerBlobObjectListPtr& outputs);

 private:
  std::shared_ptr<const ArgTuple> input_arg_tuple_;
  std::shared_ptr<const ArgTuple> output_arg_tuple_;
  std::vector<std::unique_ptr<EagerBlobObjectTensorView>> input_tensor_views_;
  std::vector<std::unique_ptr<EagerBlobObjectTensorView>> output_tensor_views_;
  std::vector<std::unique_ptr<EagerBlobObjectTensorDescView>> input_tensor_desc_views_;
  std::vector<std::unique_ptr<EagerBlobObjectTensorDescView>> output_tensor_desc_views_;
  std::unique_ptr<EagerBlobObjectTensorView> tmp_buffer_view_;
  EagerBlobObjectListPtr input_tensors_;
  EagerBlobObjectListPtr output_tensors_;
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
  const JobDesc& job_desc() const { UNIMPLEMENTED(); }

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
  }
  user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                  int32_t index) override;
  Shape* Shape4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return NonNullTensorDesc4ArgNameAndIndex(arg_name, index)->mut_shape();
  }
  DataType* Dtype4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return NonNullTensorDesc4ArgNameAndIndex(arg_name, index)->mut_data_type();
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
  const ParallelContext& parallel_ctx() const override { UNIMPLEMENTED(); };
  const ParallelDesc& parallel_desc() const override { UNIMPLEMENTED(); }
  const SbpParallel& SbpParallel4ArgNameAndIndex(const std::string& arg_name,
                                                 int32_t index) const override {
    UNIMPLEMENTED();
  }
  const ParallelDistribution& ParallelDistribution4ArgNameAndIndex(const std::string& arg_name,
                                                                   int32_t index) const override {
    UNIMPLEMENTED();
  }

  int64_t parallel_num() const override { return 1; }

  void Update(const EagerBlobObjectListPtr& inputs, const EagerBlobObjectListPtr& outputs);

 private:
  user_op::TensorDesc* NonNullTensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                         int32_t index) {
    user_op::TensorDesc* tensor_desc = TensorDesc4ArgNameAndIndex(arg_name, index);
    if (!tensor_desc) { LOG(FATAL) << "Arg (" << arg_name << "," << index << ") is not found"; }
    return tensor_desc;
  }
  const user_op::UserOpConfWrapper& user_op_conf() const override { return *user_op_conf_; }
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
  DeviceCtx* device_ctx() override { return device_ctx_; }

  DeviceType device_type() const override { return base_ctx_.device_type(); }
  const ParallelContext& parallel_ctx() const override { UNIMPLEMENTED(); };
  const JobDesc& job_desc() const override { UNIMPLEMENTED(); };

  const ArgVec& inputs() const override { return base_ctx_.inputs(); };
  const ArgVec& outputs() const override { return base_ctx_.outputs(); };

  void Update(const EagerBlobObjectListPtr& inputs, const EagerBlobObjectListPtr& outputs,
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
                                          const AttrMap& base_attrs,
                                          const std::shared_ptr<MemoryCase>& mem_case,
                                          const std::shared_ptr<const ParallelDesc>& parallel_desc,
                                          const std::shared_ptr<const ArgTuple>& input_arg_tuple,
                                          const std::shared_ptr<const ArgTuple>& output_arg_tuple);
  ~StatefulLocalOpKernel();
  const std::shared_ptr<MemoryCase> mem_case() const { return mem_case_; };
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

  std::shared_ptr<VmLocalDepObject> infer_local_dep_object() const {
    return infer_local_dep_object_;
  }
  std::shared_ptr<VmLocalDepObject> compute_local_dep_object() const {
    return compute_local_dep_object_;
  }

  Maybe<void> InferTensorDesc(const EagerBlobObjectListPtr& inputs,
                              const EagerBlobObjectListPtr& outputs,
                              LocalUserOpInferContext* op_infer_ctx);
  Maybe<void> InferDataType(const EagerBlobObjectListPtr& inputs,
                            const EagerBlobObjectListPtr& outputs,
                            LocalUserOpInferContext* op_infer_ctx);

  void ResetDynamicOpAttrs(const AttrMap& attrs);

  LocalUserOpInferContext* op_infer_ctx_for_thread_a() const {
    return op_infer_ctx_for_thread_a_.get();
  }

  LocalUserOpInferContext* op_infer_ctx_for_thread_b() const {
    return op_infer_ctx_for_thread_b_.get();
  }

 private:
  friend struct vm::LocalCallOpKernelUtil;
  StatefulLocalOpKernel() = default;
  LocalUserKernelComputeContext* UpdateComputeContext(const EagerBlobObjectListPtr& inputs,
                                                      const EagerBlobObjectListPtr& outputs,
                                                      DeviceCtx* device_ctx);

  user_op::TensorDescInferFn TensorDescInferFn() const;
  user_op::DataTypeInferFn DataTypeInferFn() const;

  void TryInitOpKernelState(const user_op::OpKernel* op_kernel, DeviceCtx* device_ctx,
                            const EagerBlobObjectListPtr& inputs,
                            const EagerBlobObjectListPtr& outputs, user_op::OpKernelState** state);

  vm::EagerBlobObject* mut_temp_blob_object();

  user_op::OpKernelState* mut_opkernel_state(const user_op::OpKernel* opkernel) {
    return op_kernel_state_map_.at(opkernel).get();
  }

  bool need_check_mem_case() const { return need_check_mem_case_; }
  void set_need_check_mem_case(bool value) { need_check_mem_case_ = value; }

  Maybe<const user_op::OpKernel*> ChooseOpKernel(const EagerBlobObjectListPtr& inputs,
                                                 const EagerBlobObjectListPtr& outputs);

  const user_op::InferTmpSizeFn& GetInferTmpSizeFn(const user_op::OpKernel* op_kernel) const;

  std::shared_ptr<OperatorConf> op_conf_;
  std::unique_ptr<ComposedAttrMap> composed_attrs_;
  std::unique_ptr<user_op::UserOpConfWrapper> user_op_conf_;
  std::shared_ptr<MemoryCase> mem_case_;
  std::unique_ptr<LocalUserKernelRegContext> reg_ctx_;
  std::unique_ptr<LocalUserKernelCreateContext> create_ctx_;
  std::unique_ptr<LocalUserOpInferContext> op_infer_ctx_for_thread_a_;
  std::unique_ptr<LocalUserOpInferContext> op_infer_ctx_for_thread_b_;
  std::unique_ptr<LocalUserKernelComputeContext> compute_ctx_;
  std::shared_ptr<const ArgTuple> input_arg_tuple_;
  std::shared_ptr<const ArgTuple> output_arg_tuple_;
  bool need_check_mem_case_;
  user_op::TensorDescInferFn tensor_desc_infer_fn_;
  user_op::DataTypeInferFn data_type_infer_fn_;
  HashMap<const user_op::OpKernelRegistryResult*, std::shared_ptr<const user_op::OpKernel>>
      op_kernel_map_;
  HashMap<const user_op::OpKernel*, std::shared_ptr<user_op::OpKernelState>> op_kernel_state_map_;
  HashMap<const user_op::OpKernel*, const user_op::InferTmpSizeFn*> infer_tmp_size_fn_map_;
  std::unique_ptr<vm::EagerBlobObject> tmp_blob_object_;
  std::vector<int64_t> input_tuple_indexes4const_ibns_;
  std::vector<int64_t> input_tuple_indexes4mut_ibns_;
  std::vector<int64_t> output_tuple_indexes4mut_obns_;
  std::vector<int64_t> output_tuple_indexes4mut2_obns_;
  std::shared_ptr<VmLocalDepObject> infer_local_dep_object_;
  std::shared_ptr<VmLocalDepObject> compute_local_dep_object_;
};

}  // namespace one

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_STATEFUL_LOCAL_OPKERNEL_H_
