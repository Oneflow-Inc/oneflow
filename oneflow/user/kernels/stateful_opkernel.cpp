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
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/kernel/blob_tensor_view.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_helper.h"
#include "oneflow/core/eager/eager_blob_object.h"

namespace oneflow {
namespace one {

template<class T>
class TensorsPtrScope {
 public:
  TensorsPtrScope(T* ctx, TensorsPtr inputs, TensorsPtr outputs) {
    ctx_ = ctx;
    ctx_->Update(inputs, outputs);
  }
  ~TensorsPtrScope() { ctx_->Update(nullptr, nullptr); }

 private:
  T* ctx_;
};

class LocalUserKernelBaseContext {
 public:
  LocalUserKernelBaseContext(const std::string& device_tag,
                             const std::shared_ptr<const JobDesc> job_desc,
                             TensorIndexMap bn_in_op2bn_index2input_tensor_index,
                             TensorIndexMap bn_in_op2bn_index2output_tensor_index)
      : job_desc_(job_desc),
        bn_in_op2bn_index2input_tensor_index_(bn_in_op2bn_index2input_tensor_index),
        bn_in_op2bn_index2output_tensor_index_(bn_in_op2bn_index2output_tensor_index) {
    device_tag_ = device_tag;
    device_type_ = CHECK_JUST(DeviceType4DeviceTag(device_tag_));

    for (auto& pair : *bn_in_op2bn_index2input_tensor_index) {
      const auto& bn_in_op = pair.first;
      for (int64_t bn_index = 0; bn_index < pair.second.size(); bn_index++) {
        inputs_.push_back({bn_in_op, bn_index});
      }
    }
    for (auto& pair : *bn_in_op2bn_index2output_tensor_index) {
      const auto& bn_in_op = pair.first;
      for (int64_t bn_index = 0; bn_index < pair.second.size(); bn_index++) {
        outputs_.push_back({bn_in_op, bn_index});
      }
    }
  }
  ~LocalUserKernelBaseContext() = default;

  DeviceType device_type() const { return device_type_; }
  const std::string& device_tag() const { return device_tag_; }
  const JobDesc& job_desc() const { return *job_desc_; }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const {
    auto it = arg2tensor_desc_.find(std::make_pair(arg_name, index));
    CHECK(it != arg2tensor_desc_.end()) << "Arg (" << arg_name << "," << index << ") is not found";
    return it->second.get();
  }

  user_op::Tensor* Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t index) const {
    auto it = arg2tensor_.find(std::make_pair(arg_name, index));
    CHECK(it != arg2tensor_.end()) << "Arg (" << arg_name << "," << index << ") is not found";
    return it->second.get();
  }

  void Update(TensorsPtr inputs, TensorsPtr outputs) {
    input_tensors_ = inputs;
    output_tensors_ = outputs;

    arg2tensor_.clear();
    arg2tensor_desc_.clear();

    auto UpdateArg2TensorAndTensorDesc = [this](TensorsPtr tensors_ptr,
                                                TensorIndexMap tensor_index_map) {
      if (!tensors_ptr) { return; }
      for (auto& pair : *tensor_index_map) {
        const auto& bn_in_op = pair.first;
        for (int64_t bn_index = 0; bn_index < pair.second.size(); bn_index++) {
          const auto tensor_index = pair.second[bn_index];
          const auto tensor = (*tensors_ptr)[tensor_index];
          arg2tensor_.emplace(std::make_pair(bn_in_op, bn_index),
                              std::make_shared<user_op::BlobTensorView>(tensor->mut_blob()));
          arg2tensor_desc_.emplace(std::make_pair(bn_in_op, bn_index),
                                   std::make_shared<EagerBlobObjectTensorDescView>(tensor));
        }
      }
    };

    UpdateArg2TensorAndTensorDesc(input_tensors_, bn_in_op2bn_index2input_tensor_index_);
    UpdateArg2TensorAndTensorDesc(output_tensors_, bn_in_op2bn_index2output_tensor_index_);
  }

  const ArgVec& inputs() const { return inputs_; }
  const ArgVec& outputs() const { return outputs_; }

 private:
  ArgVec inputs_;
  ArgVec outputs_;
  DeviceType device_type_;
  std::string device_tag_;
  std::shared_ptr<const JobDesc> job_desc_;
  TensorIndexMap bn_in_op2bn_index2input_tensor_index_;
  TensorIndexMap bn_in_op2bn_index2output_tensor_index_;
  HashMap<std::pair<std::string, int64_t>, std::shared_ptr<user_op::BlobTensorView>> arg2tensor_;
  HashMap<std::pair<std::string, int64_t>, std::shared_ptr<EagerBlobObjectTensorDescView>>
      arg2tensor_desc_;
  TensorsPtr input_tensors_;
  TensorsPtr output_tensors_;
};

class LocalUserKernelRegContext final : public user_op::KernelRegContext {
 public:
  explicit LocalUserKernelRegContext(const OperatorConf& op_conf,
                                     const std::shared_ptr<const JobDesc> job_desc,
                                     TensorIndexMap bn_in_op2bn_index2input_tensor_index,
                                     TensorIndexMap bn_in_op2bn_index2output_tensor_index)
      : user_op::KernelRegContext(user_op::UserOpConfWrapper(op_conf)),
        base_ctx_(LocalUserKernelBaseContext(op_conf.device_tag(), job_desc,
                                             bn_in_op2bn_index2input_tensor_index,
                                             bn_in_op2bn_index2output_tensor_index)) {}
  ~LocalUserKernelRegContext() = default;

  DeviceType device_type() const override { return base_ctx_.device_type(); }
  const std::string& device_tag() const override { return base_ctx_.device_tag(); }
  const ParallelContext& parallel_ctx() const override { UNIMPLEMENTED(); }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return base_ctx_.TensorDesc4ArgNameAndIndex(arg_name, index);
  }
  const ArgVec& inputs() const override { return base_ctx_.inputs(); }
  const ArgVec& outputs() const override { return base_ctx_.outputs(); }

  void Update(TensorsPtr inputs, TensorsPtr outputs) { base_ctx_.Update(inputs, outputs); }

 private:
  LocalUserKernelBaseContext base_ctx_;
};

class LocalUserKernelCreateContext final : public user_op::KernelCreateContext {
 public:
  explicit LocalUserKernelCreateContext(const OperatorConf& op_conf) : user_op_conf_(op_conf) {}

  const user_op::UserOpConfWrapper& user_op_conf() const override { return user_op_conf_; }

 private:
  user_op::UserOpConfWrapper user_op_conf_;
};

class LocalUserKernelInitContext final : public user_op::KernelInitContext {
 public:
  explicit LocalUserKernelInitContext(DeviceCtx* device_ctx, const OperatorConf& op_conf,
                                      const std::shared_ptr<const JobDesc> job_desc,
                                      TensorIndexMap bn_in_op2bn_index2input_tensor_index,
                                      TensorIndexMap bn_in_op2bn_index2output_tensor_index)
      : user_op::KernelInitContext(user_op::UserOpConfWrapper(op_conf)),
        device_ctx_(device_ctx),
        base_ctx_(LocalUserKernelBaseContext(op_conf.device_tag(), job_desc,
                                             bn_in_op2bn_index2input_tensor_index,
                                             bn_in_op2bn_index2output_tensor_index)) {}
  ~LocalUserKernelInitContext() override = default;

  void set_device_ctx(DeviceCtx* device_ctx) { device_ctx_ = device_ctx; }
  DeviceCtx* device_ctx() override { return device_ctx_; }

  DeviceType device_type() const override { return base_ctx_.device_type(); }
  const ParallelContext& parallel_ctx() const override { UNIMPLEMENTED(); }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return base_ctx_.TensorDesc4ArgNameAndIndex(arg_name, index);
  }
  const user_op::TensorDesc* LogicalTensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                               int32_t index) const override {
    UNIMPLEMENTED();
  }
  const SbpParallel& SbpParallel4ArgNameAndIndex(const std::string& arg_name,
                                                 int32_t index) const override {
    UNIMPLEMENTED();
  }

  const ParallelDistribution& ParallelDistribution4ArgNameAndIndex(const std::string& arg_name,
                                                                   int32_t index) const override {
    UNIMPLEMENTED();
  }

  const ArgVec& inputs() const override { return base_ctx_.inputs(); }
  const ArgVec& outputs() const override { return base_ctx_.outputs(); }
  const ParallelDesc& parallel_desc() const override { UNIMPLEMENTED(); }

  void Update(TensorsPtr inputs, TensorsPtr outputs) { base_ctx_.Update(inputs, outputs); }

 private:
  DeviceCtx* device_ctx_;
  LocalUserKernelBaseContext base_ctx_;
};

LocalUserOpInferContext::LocalUserOpInferContext(
    const OperatorConf& op_conf, const std::shared_ptr<const JobDesc> job_desc,
    TensorIndexMap bn_in_op2bn_index2input_tensor_index,
    TensorIndexMap bn_in_op2bn_index2output_tensor_index)
    : user_op_conf_(op_conf),
      job_desc_(job_desc),
      bn_in_op2bn_index2input_tensor_index_(bn_in_op2bn_index2input_tensor_index),
      bn_in_op2bn_index2output_tensor_index_(bn_in_op2bn_index2output_tensor_index) {
  for (auto& pair : *bn_in_op2bn_index2input_tensor_index) {
    const auto& bn_in_op = pair.first;
    for (int64_t bn_index = 0; bn_index < pair.second.size(); bn_index++) {
      inputs_.push_back({bn_in_op, bn_index});
    }
  }
  for (auto& pair : *bn_in_op2bn_index2output_tensor_index) {
    const auto& bn_in_op = pair.first;
    for (int64_t bn_index = 0; bn_index < pair.second.size(); bn_index++) {
      outputs_.push_back({bn_in_op, bn_index});
    }
  }
}

user_op::TensorDesc* LocalUserOpInferContext::TensorDesc4ArgNameAndIndex(
    const std::string& arg_name, int32_t index) {
  auto it = arg2tensor_desc_.find(std::make_pair(arg_name, index));
  CHECK(it != arg2tensor_desc_.end()) << "Arg (" << arg_name << "," << index << ") is not found";
  return it->second.get();
}

void LocalUserOpInferContext::Update(TensorsPtr inputs, TensorsPtr outputs) {
  input_tensors_ = inputs;
  output_tensors_ = outputs;

  arg2tensor_desc_.clear();
  auto UpdateArg2TensorDesc = [this](TensorsPtr tensors_ptr, TensorIndexMap tensor_index_map) {
    if (!tensors_ptr) { return; }
    for (auto& pair : *tensor_index_map) {
      const auto& bn_in_op = pair.first;
      for (int64_t bn_index = 0; bn_index < pair.second.size(); bn_index++) {
        const auto tensor_index = pair.second[bn_index];
        const auto tensor = (*tensors_ptr)[tensor_index];
        arg2tensor_desc_.emplace(std::make_pair(bn_in_op, bn_index),
                                 std::make_shared<EagerBlobObjectTensorDescView>(tensor));
      }
    }
  };

  UpdateArg2TensorDesc(input_tensors_, bn_in_op2bn_index2input_tensor_index_);
  UpdateArg2TensorDesc(output_tensors_, bn_in_op2bn_index2output_tensor_index_);
}

LocalUserKernelComputeContext::LocalUserKernelComputeContext(
    DeviceCtx* device_ctx, const OperatorConf& op_conf,
    const std::shared_ptr<const JobDesc> job_desc,
    TensorIndexMap bn_in_op2bn_index2input_tensor_index,
    TensorIndexMap bn_in_op2bn_index2output_tensor_index)
    : user_op::KernelComputeContext(user_op::UserOpConfWrapper(op_conf)),
      device_ctx_(device_ctx),
      base_ctx_(std::unique_ptr<LocalUserKernelBaseContext>(new LocalUserKernelBaseContext(
          op_conf.device_tag(), job_desc, bn_in_op2bn_index2input_tensor_index,
          bn_in_op2bn_index2output_tensor_index))) {}

const user_op::TensorDesc* LocalUserKernelComputeContext::TensorDesc4ArgNameAndIndex(
    const std::string& arg_name, int32_t index) const {
  return base_ctx_->TensorDesc4ArgNameAndIndex(arg_name, index);
}

user_op::Tensor* LocalUserKernelComputeContext::Tensor4ArgNameAndIndex(const std::string& arg_name,
                                                                       int32_t index) {
  return base_ctx_->Tensor4ArgNameAndIndex(arg_name, index);
}
DeviceCtx* LocalUserKernelComputeContext::device_ctx() { return device_ctx_; }

DeviceType LocalUserKernelComputeContext::device_type() const { return base_ctx_->device_type(); }
const ParallelContext& LocalUserKernelComputeContext::parallel_ctx() const { UNIMPLEMENTED(); }
const JobDesc& LocalUserKernelComputeContext::job_desc() const { return base_ctx_->job_desc(); }

const ArgVec& LocalUserKernelComputeContext::inputs() const { return base_ctx_->inputs(); }
const ArgVec& LocalUserKernelComputeContext::outputs() const { return base_ctx_->outputs(); }

void LocalUserKernelComputeContext::Update(TensorsPtr inputs, TensorsPtr outputs,
                                           DeviceCtx* device_ctx) {
  input_tensors_ = inputs;
  output_tensors_ = outputs;
  device_ctx_ = device_ctx;
  base_ctx_->Update(inputs, outputs);
}

StatefulOpKernel::StatefulOpKernel(const std::shared_ptr<const JobDesc> job_desc,
                                   const OperatorConf& op_conf,
                                   const std::shared_ptr<MemoryCase>& mem_case,
                                   TensorIndexMap bn_in_op2bn_index2input_tensor_index,
                                   TensorIndexMap bn_in_op2bn_index2output_tensor_index)
    : job_desc_(job_desc),
      op_conf_(op_conf),
      mem_case_(mem_case),
      bn_in_op2bn_index2input_tensor_index_(bn_in_op2bn_index2input_tensor_index),
      bn_in_op2bn_index2output_tensor_index_(bn_in_op2bn_index2output_tensor_index) {
  op_infer_ctx_.reset(new LocalUserOpInferContext(op_conf, job_desc,
                                                  bn_in_op2bn_index2input_tensor_index,
                                                  bn_in_op2bn_index2output_tensor_index));
  compute_ctx_.reset(new LocalUserKernelComputeContext(nullptr, op_conf, job_desc,
                                                       bn_in_op2bn_index2input_tensor_index,
                                                       bn_in_op2bn_index2output_tensor_index));
  create_ctx_.reset(new LocalUserKernelCreateContext(op_conf));
  reg_ctx_.reset(new LocalUserKernelRegContext(op_conf, job_desc,
                                               bn_in_op2bn_index2input_tensor_index_,
                                               bn_in_op2bn_index2output_tensor_index_));
  const auto* op_reg_val =
      user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_conf.user_conf().op_type_name());
  CHECK_NOTNULL(op_reg_val);
  if (op_reg_val->physical_tensor_desc_infer_fn) {
    tensor_desc_infer_fn_ = op_reg_val->physical_tensor_desc_infer_fn;
  } else {
    UNIMPLEMENTED();
  }

  DataType default_data_type = job_desc->DefaultDataType();
  CHECK_NE(default_data_type, DataType::kInvalidDataType);
  tmp_blob_object_.reset(new eager::EagerBlobObject(mem_case_, std::make_shared<Shape>(),
                                                    default_data_type,
                                                    std::make_shared<eager::TensorBuffer>()));
}

Maybe<const user_op::OpKernel*> StatefulOpKernel::GetOpKernel(TensorsPtr inputs,
                                                              TensorsPtr outputs) {
  TensorsPtrScope<LocalUserKernelRegContext> reg_ctx_scope(reg_ctx_.get(), inputs, outputs);
  const auto& op_type_name = op_conf_.user_conf().op_type_name();
  const auto* kernel_reg_val = CHECK_JUST(
      user_op::UserOpRegistryMgr::Get().GetOpKernelRegistryResult(op_type_name, *reg_ctx_));
  CHECK_NOTNULL(kernel_reg_val);
  auto it = op_kernel_map_.find(kernel_reg_val);
  if (it != op_kernel_map_.end()) { return it->second.get(); }

  auto* kernel = kernel_reg_val->create_fn(create_ctx_.get());
  op_kernel_map_.emplace(
      std::make_pair(kernel_reg_val, std::shared_ptr<const user_op::OpKernel>(kernel)));
  auto init_ctx = std::make_shared<LocalUserKernelInitContext>(
      nullptr, op_conf_, job_desc_, bn_in_op2bn_index2input_tensor_index_,
      bn_in_op2bn_index2output_tensor_index_);
  init_ctx->Update(inputs, outputs);
  init_ctx_map_.emplace(std::make_pair(kernel, init_ctx));

  infer_tmp_size_fn_map_.emplace(std::make_pair(kernel, &kernel_reg_val->infer_tmp_size_fn));

  return kernel;
}

void StatefulOpKernel::ChooseOpKernel(TensorsPtr inputs, TensorsPtr outputs) {
  TensorsPtrScope<LocalUserKernelRegContext> reg_ctx_scope(reg_ctx_.get(), inputs, outputs);
  const auto& op_type_name = op_conf_.user_conf().op_type_name();
  const auto* kernel_reg_val = CHECK_JUST(
      user_op::UserOpRegistryMgr::Get().GetOpKernelRegistryResult(op_type_name, *reg_ctx_));
  CHECK_NOTNULL(kernel_reg_val);
  auto it = op_kernel_map_.find(kernel_reg_val);
  if (it != op_kernel_map_.end()) { current_op_kernel_ = it->second.get(); }

  auto* kernel = kernel_reg_val->create_fn(create_ctx_.get());
  op_kernel_map_.emplace(
      std::make_pair(kernel_reg_val, std::shared_ptr<const user_op::OpKernel>(kernel)));
  auto init_ctx = std::make_shared<LocalUserKernelInitContext>(
      nullptr, op_conf_, job_desc_, bn_in_op2bn_index2input_tensor_index_,
      bn_in_op2bn_index2output_tensor_index_);
  init_ctx->Update(inputs, outputs);
  init_ctx_map_.emplace(std::make_pair(kernel, init_ctx));

  infer_tmp_size_fn_map_.emplace(std::make_pair(kernel, &kernel_reg_val->infer_tmp_size_fn));

  current_op_kernel_ = kernel;
}

Maybe<user_op::OpKernelState> StatefulOpKernel::TryInitOpKernelState(DeviceCtx* device_ctx) {
  auto it = op_kernel_state_map_.find(current_op_kernel_);
  if (it != op_kernel_state_map_.end()) { return it->second; }

  auto init_ctx = init_ctx_map_.at(current_op_kernel_);
  init_ctx->set_device_ctx(device_ctx);
  auto state = kernel_->CreateOpKernelState(init_ctx.get());
  op_kernel_state_map_.emplace(std::make_pair(current_op_kernel_, state));
  init_ctx->set_device_ctx(nullptr);
  return state;
}

const user_op::InferTmpSizeFn& StatefulOpKernel::GetInferTmpSizeFn() const {
  return *infer_tmp_size_fn_map_.at(current_op_kernel_);
}

eager::EagerBlobObject* StatefulOpKernel::mut_temp_blob_object() { return tmp_blob_object_.get(); }

user_op::TensorDescInferFn StatefulOpKernel::TensorDescInferFn() const {
  return tensor_desc_infer_fn_;
}

LocalUserOpInferContext* StatefulOpKernel::UpdateInferContext(TensorsPtr inputs,
                                                              TensorsPtr outputs) {
  op_infer_ctx_->Update(inputs, outputs);
  return op_infer_ctx_.get();
}

LocalUserKernelComputeContext* StatefulOpKernel::UpdateComputeContext(TensorsPtr inputs,
                                                                      TensorsPtr outputs,
                                                                      DeviceCtx* device_ctx) {
  compute_ctx_->Update(inputs, outputs, device_ctx);
  return compute_ctx_.get();
}

}  // namespace one
}  // namespace oneflow
