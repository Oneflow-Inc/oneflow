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

using TensorsPtr = std::vector<std::shared_ptr<eager::EagerBlobObject>>*;
using TensorIndexMap = std::shared_ptr<HashMap<std::string, std::vector<int64_t>>>;
using OpKernelMap =
    HashMap<const user_op::OpKernelRegistryResult*, std::shared_ptr<const user_op::OpKernel>>;
using InitCtxMap = HashMap<const user_op::OpKernel*, std::shared_ptr<LocalUserKernelInitContext>>;
using OpKernelStateMap = HashMap<const user_op::OpKernel*, std::shared_ptr<user_op::OpKernelState>>;
using InferTmpSizeFnMap = HashMap<const user_op::OpKernel*, const user_op::InferTmpSizeFn*>;

class EagerBlobObjectTensorDescView final : public user_op::TensorDesc {
 public:
  EagerBlobObjectTensorDescView(std::shared_ptr<eager::EagerBlobObject> blob_object) {
    blob_object_ = blob_object;
  }

  const Shape& shape() const override { return blob_object_->blob_desc().shape(); }

  Shape* mut_shape() override { return &blob_object_->mut_blob_desc()->mut_shape(); }

  DataType data_type() const override { return blob_object_->blob_desc().data_type(); }

  DataType* mut_data_type() override { return blob_object_->mut_blob_desc()->mut_data_type(); }

  bool is_dynamic() const override { return blob_object_->blob_desc().is_dynamic(); }
  bool* mut_is_dynamic() override { return blob_object_->mut_blob_desc()->mut_is_dynamic(); }
  void set_is_dynamic(bool val) override { blob_object_->mut_blob_desc()->set_is_dynamic(val); }

 private:
  std::shared_ptr<eager::EagerBlobObject> blob_object_;
};

class LocalUserOpInferContext : public user_op::InferContext {
 public:
  LocalUserOpInferContext(const OperatorConf& op_conf,
                          const std::shared_ptr<const JobDesc> job_desc,
                          TensorIndexMap bn_in_op2bn_index2input_tensor_index,
                          TensorIndexMap bn_in_op2bn_index2output_tensor_index);
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

  const ArgVec& inputs() const override { return inputs_; }
  const ArgVec& outputs() const override { return outputs_; }
  const JobDesc* job_desc() const override {
    CHECK_NOTNULL(job_desc_);
    return job_desc_.get();
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

  void Update(TensorsPtr inputs, TensorsPtr outputs);

 private:
  user_op::UserOpConfWrapper user_op_conf_;
  std::shared_ptr<const JobDesc> job_desc_;
  ArgVec inputs_;
  ArgVec outputs_;
  TensorIndexMap bn_in_op2bn_index2input_tensor_index_;
  TensorIndexMap bn_in_op2bn_index2output_tensor_index_;
  TensorsPtr input_tensors_;
  TensorsPtr output_tensors_;
  HashMap<std::pair<std::string, int64_t>, std::shared_ptr<EagerBlobObjectTensorDescView>>
      arg2tensor_desc_;
};

class LocalUserKernelComputeContext final : public user_op::KernelComputeContext {
 public:
  explicit LocalUserKernelComputeContext(DeviceCtx* device_ctx, const OperatorConf& op_conf,
                                         const std::shared_ptr<const JobDesc> job_desc,
                                         TensorIndexMap bn_in_op2bn_index2input_tensor_index,
                                         TensorIndexMap bn_in_op2bn_index2output_tensor_index);
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

  void Update(TensorsPtr inputs, TensorsPtr outputs, DeviceCtx* device_ctx);

 private:
  DeviceCtx* device_ctx_;
  std::unique_ptr<LocalUserKernelBaseContext> base_ctx_;
  TensorIndexMap bn_in_op2bn_index2input_tensor_index_;
  TensorIndexMap bn_in_op2bn_index2output_tensor_index_;
  TensorsPtr input_tensors_;
  TensorsPtr output_tensors_;
};

class StatefulOpKernel final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StatefulOpKernel);
  StatefulOpKernel(const std::shared_ptr<const JobDesc> job_desc, const OperatorConf& op_conf,
                   const std::shared_ptr<MemoryCase>& mem_case,
                   TensorIndexMap bn_in_op2bn_index2input_tensor_index,
                   TensorIndexMap bn_in_op2bn_index2output_tensor_index);
  ~StatefulOpKernel() = default;

  LocalUserOpInferContext* UpdateInferContext(TensorsPtr inputs, TensorsPtr outputs);
  LocalUserKernelComputeContext* UpdateComputeContext(TensorsPtr inputs, TensorsPtr outputs,
                                                      DeviceCtx* device_ctx);

  user_op::TensorDescInferFn TensorDescInferFn() const;

  Maybe<user_op::OpKernelState> TryInitOpKernelState(DeviceCtx* device_ctx);

  eager::EagerBlobObject* mut_temp_blob_object();

  user_op::OpKernelState* mut_opkernel_state() { return state_.get(); }

  const std::shared_ptr<MemoryCase> mem_case() const { return mem_case_; };
  bool need_check_mem_case() const { return need_check_mem_case_; }
  void set_need_check_mem_case(bool value) { need_check_mem_case_ = value; }

  Maybe<const user_op::OpKernel*> GetOpKernel(TensorsPtr inputs, TensorsPtr outputs);

  void ChooseOpKernel(TensorsPtr inputs, TensorsPtr outputs);

  const user_op::InferTmpSizeFn& GetInferTmpSizeFn() const;

  const user_op::OpKernel* mut_user_opkernel() { return current_op_kernel_; }

 private:
  friend class eager::LocalCallOpKernelUtil;
  std::shared_ptr<const JobDesc> job_desc_;
  OperatorConf op_conf_;
  std::shared_ptr<MemoryCase> mem_case_;
  std::unique_ptr<const user_op::OpKernel> kernel_;
  std::unique_ptr<LocalUserKernelRegContext> reg_ctx_;
  std::unique_ptr<LocalUserKernelCreateContext> create_ctx_;
  std::unique_ptr<LocalUserOpInferContext> op_infer_ctx_;
  std::unique_ptr<LocalUserKernelComputeContext> compute_ctx_;
  std::shared_ptr<user_op::OpKernelState> state_;
  TensorIndexMap bn_in_op2bn_index2input_tensor_index_;
  TensorIndexMap bn_in_op2bn_index2output_tensor_index_;
  bool need_check_mem_case_;
  user_op::TensorDescInferFn tensor_desc_infer_fn_;
  OpKernelMap op_kernel_map_;
  OpKernelStateMap op_kernel_state_map_;
  InitCtxMap init_ctx_map_;
  InferTmpSizeFnMap infer_tmp_size_fn_map_;
  std::unique_ptr<eager::EagerBlobObject> tmp_blob_object_;
  const user_op::OpKernel* current_op_kernel_;
};

}  // namespace one

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_STATEFUL_OPKERNEL_H_
