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
#ifndef ONEFLOW_USER_KERNELS_STATEFUL_OPKERNEL_H_
#define ONEFLOW_USER_KERNELS_STATEFUL_OPKERNEL_H_

#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/user_op_kernel_registry.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

namespace eager {
struct LocalCallOpKernelUtil;
class EagerBlobObject;
}  // namespace eager

namespace one {

class LocalUserKernelBaseContext;
class LocalUserKernelRegContext;
class LocalUserKernelCreateContext;
class LocalUserKernelInitContext;
class LocalUserOpInferContext;

using ArgVec = std::vector<std::pair<std::string, int32_t>>;

using EagerBlobObjectList = std::shared_ptr<std::vector<std::shared_ptr<eager::EagerBlobObject>>>;

class LocalUserOpArgContext {
 public:
  LocalUserOpArgContext(std::function<eager::EagerBlobObject*(int64_t)> getter) : getter_(getter){};
  eager::EagerBlobObject* MutEagerBlobObject(int64_t index) const { return getter_(index); }

 private:
  std::function<eager::EagerBlobObject*(int64_t)> getter_;
};

class EagerBlobObjectTensorView final : public user_op::Tensor {
 public:
  EagerBlobObjectTensorView(LocalUserOpArgContext* ctx, int64_t index) : ctx_(ctx), index_(index) {}

  const ShapeView& shape() const override {
    return ctx_->MutEagerBlobObject(index_)->blob().shape();
  }

  MutShapeView* mut_shape() override {
    return ctx_->MutEagerBlobObject(index_)->mut_blob()->mut_shape_view();
  }

  DataType data_type() const override {
    return ctx_->MutEagerBlobObject(index_)->blob().data_type();
  }

  const MemoryCase& mem_case() const override {
    return ctx_->MutEagerBlobObject(index_)->blob().mem_case();
  }

  const void* raw_dptr() const override { return ctx_->MutEagerBlobObject(index_)->blob().dptr(); }

  void* mut_raw_dptr() override { return ctx_->MutEagerBlobObject(index_)->mut_blob()->mut_dptr(); }

 private:
  LocalUserOpArgContext* ctx_;
  const int64_t index_;
};

class EagerBlobObjectTensorDescView final : public user_op::TensorDesc {
 public:
  EagerBlobObjectTensorDescView(LocalUserOpArgContext* ctx, int64_t index)
      : ctx_(ctx), index_(index) {}

  const Shape& shape() const override {
    return ctx_->MutEagerBlobObject(index_)->blob_desc().shape();
  }

  Shape* mut_shape() override {
    return &ctx_->MutEagerBlobObject(index_)->mut_blob_desc()->mut_shape();
  }

  DataType data_type() const override {
    return ctx_->MutEagerBlobObject(index_)->blob_desc().data_type();
  }

  DataType* mut_data_type() override {
    return ctx_->MutEagerBlobObject(index_)->mut_blob_desc()->mut_data_type();
  }

  bool is_dynamic() const override {
    return ctx_->MutEagerBlobObject(index_)->blob_desc().is_dynamic();
  }

  bool* mut_is_dynamic() override {
    return ctx_->MutEagerBlobObject(index_)->mut_blob_desc()->mut_is_dynamic();
  }

  void set_is_dynamic(bool val) override {
    ctx_->MutEagerBlobObject(index_)->mut_blob_desc()->set_is_dynamic(val);
  }

 private:
  LocalUserOpArgContext* ctx_;
  const int64_t index_;
};

class LocalUserOpInferContext : public user_op::InferContext {
 public:
  LocalUserOpInferContext(const OperatorConf& op_conf, const ArgVec* indexed_input_pairs,
                          const ArgVec* indexed_output_pairs);
  ~LocalUserOpInferContext() override = default;

  const user_op::TensorDesc* LogicalTensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                               int32_t index) const override {
    UNIMPLEMENTED();
  }
  user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                  int32_t index) override;
  Shape* Shape4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return TensorDesc4ArgNameAndIndex(arg_name, index)->mut_shape();
  }
  DataType* Dtype4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return TensorDesc4ArgNameAndIndex(arg_name, index)->mut_data_type();
  }
  bool* IsDynamic4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return TensorDesc4ArgNameAndIndex(arg_name, index)->mut_is_dynamic();
  }

  const ArgVec& inputs() const override { return *indexed_input_pairs_; }
  const ArgVec& outputs() const override { return *indexed_output_pairs_; }
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

  const user_op::UserOpConfWrapper& user_op_conf() const override { return user_op_conf_; }

  void Update(EagerBlobObjectList inputs, EagerBlobObjectList outputs);

 private:
  user_op::UserOpConfWrapper user_op_conf_;
  const ArgVec* indexed_input_pairs_;
  const ArgVec* indexed_output_pairs_;
  EagerBlobObjectList input_tensors_;
  EagerBlobObjectList output_tensors_;
  LocalUserOpArgContext input_arg_context_;
  LocalUserOpArgContext output_arg_context_;
  mutable std::vector<EagerBlobObjectTensorDescView> input_tensor_desc_views_;
  mutable std::vector<EagerBlobObjectTensorDescView> output_tensor_desc_views_;
};

class LocalUserKernelComputeContext final : public user_op::KernelComputeContext {
 public:
  explicit LocalUserKernelComputeContext(DeviceCtx* device_ctx, const OperatorConf& op_conf,
                                         const ArgVec* indexed_input_pairs,
                                         const ArgVec* indexed_output_pairs);
  ~LocalUserKernelComputeContext() = default;

  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override;

  user_op::Tensor* Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t index) override;
  DeviceCtx* device_ctx() override;

  DeviceType device_type() const override;
  const ParallelContext& parallel_ctx() const override;
  const JobDesc& job_desc() const override;

  const ArgVec& inputs() const override;
  const ArgVec& outputs() const override;

  void Update(EagerBlobObjectList inputs, EagerBlobObjectList outputs, DeviceCtx* device_ctx);

 private:
  DeviceCtx* device_ctx_;
  std::unique_ptr<LocalUserKernelBaseContext> base_ctx_;
  const ArgVec* indexed_input_pairs_;
  const ArgVec* indexed_output_pairs_;
  EagerBlobObjectList input_tensors_;
  EagerBlobObjectList output_tensors_;
};

class StatefulOpKernel final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StatefulOpKernel);
  StatefulOpKernel(const OperatorConf& op_conf, const std::shared_ptr<MemoryCase>& mem_case,
                   const ArgVec* indexed_input_pairs, const ArgVec* indexed_output_pairs);
  ~StatefulOpKernel();
  const std::shared_ptr<MemoryCase> mem_case() const { return mem_case_; };
  Maybe<void> set_device(const DeviceType dev_type, const int64_t dev_id,
                         const std::string& dev_tag);

 private:
  friend struct eager::LocalCallOpKernelUtil;
  LocalUserOpInferContext* UpdateInferContext(EagerBlobObjectList inputs,
                                              EagerBlobObjectList outputs);
  LocalUserKernelComputeContext* UpdateComputeContext(EagerBlobObjectList inputs,
                                                      EagerBlobObjectList outputs,
                                                      DeviceCtx* device_ctx);

  user_op::TensorDescInferFn TensorDescInferFn() const;
  user_op::DataTypeInferFn DataTypeInferFn() const;

  Maybe<user_op::OpKernelState> TryInitOpKernelState(DeviceCtx* device_ctx);

  eager::EagerBlobObject* mut_temp_blob_object();

  user_op::OpKernelState* mut_opkernel_state() { return current_state_; }

  bool need_check_mem_case() const { return need_check_mem_case_; }
  void set_need_check_mem_case(bool value) { need_check_mem_case_ = value; }

  void ChooseOpKernel(EagerBlobObjectList inputs, EagerBlobObjectList outputs);

  const user_op::InferTmpSizeFn& GetInferTmpSizeFn() const;

  const user_op::OpKernel* mut_user_opkernel() { return current_op_kernel_; }

  OperatorConf op_conf_;
  std::shared_ptr<MemoryCase> mem_case_;
  std::unique_ptr<LocalUserKernelRegContext> reg_ctx_;
  std::unique_ptr<LocalUserKernelCreateContext> create_ctx_;
  std::unique_ptr<LocalUserOpInferContext> op_infer_ctx_;
  std::unique_ptr<LocalUserKernelComputeContext> compute_ctx_;
  const ArgVec* indexed_input_pairs_;
  const ArgVec* indexed_output_pairs_;
  bool need_check_mem_case_;
  user_op::TensorDescInferFn tensor_desc_infer_fn_;
  user_op::DataTypeInferFn data_type_infer_fn_;
  HashMap<const user_op::OpKernelRegistryResult*, std::shared_ptr<const user_op::OpKernel>>
      op_kernel_map_;
  HashMap<const user_op::OpKernel*, std::shared_ptr<user_op::OpKernelState>> op_kernel_state_map_;
  HashMap<const user_op::OpKernel*, std::shared_ptr<LocalUserKernelInitContext>> init_ctx_map_;
  HashMap<const user_op::OpKernel*, const user_op::InferTmpSizeFn*> infer_tmp_size_fn_map_;
  std::unique_ptr<eager::EagerBlobObject> tmp_blob_object_;
  const user_op::OpKernel* current_op_kernel_;
  user_op::OpKernelState* current_state_;
};

}  // namespace one

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_STATEFUL_OPKERNEL_H_
