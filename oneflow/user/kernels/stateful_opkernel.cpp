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

using ArgVec = std::vector<std::pair<std::string, int32_t>>;

class ZeroCopyTensorDesc {};

class LocalUserKernelBaseContext {
 public:
  LocalUserKernelBaseContext(const KernelConf& kernel_conf,
                             const std::shared_ptr<const JobDesc> job_desc,
                             TensorIndexMap bn_in_op2bn_index2input_tensor_index,
                             TensorIndexMap bn_in_op2bn_index2output_tensor_index)
      : job_desc_(job_desc),
        bn_in_op2bn_index2input_tensor_index_(bn_in_op2bn_index2input_tensor_index),
        bn_in_op2bn_index2output_tensor_index_(bn_in_op2bn_index2output_tensor_index) {
    CHECK(kernel_conf.has_user_conf());
    CHECK(kernel_conf.op_attribute().op_conf().has_user_conf());

    device_tag_ = kernel_conf.op_attribute().op_conf().device_tag();
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
    // return it->second.get();
    return nullptr;
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
      for (auto& pair : *tensor_index_map) {
        const auto& bn_in_op = pair.first;
        for (int64_t bn_index = 0; bn_index < pair.second.size(); bn_index++) {
          const auto tensor_index = pair.second[bn_index];
          arg2tensor_.emplace(
              std::make_pair(bn_in_op, bn_index),
              std::make_shared<user_op::BlobTensorView>((*tensors_ptr)[tensor_index]->mut_blob()));
          // TODO: also update arg2tensor_desc_
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
  HashMap<std::pair<std::string, int64_t>, std::shared_ptr<ZeroCopyTensorDesc>> arg2tensor_desc_;
  TensorsPtr input_tensors_;
  TensorsPtr output_tensors_;
};

class LocalUserKernelRegContext final : public user_op::KernelRegContext {
 public:
  explicit LocalUserKernelRegContext(const KernelConf& kernel_conf,
                                     const std::shared_ptr<const JobDesc> job_desc,
                                     TensorIndexMap bn_in_op2bn_index2input_tensor_index,
                                     TensorIndexMap bn_in_op2bn_index2output_tensor_index)
      : user_op::KernelRegContext(user_op::UserOpConfWrapper(kernel_conf.op_attribute().op_conf())),
        base_ctx_(LocalUserKernelBaseContext(kernel_conf, job_desc,
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

 private:
  LocalUserKernelBaseContext base_ctx_;
};

class LocalUserKernelCreateContext final : public user_op::KernelCreateContext {
 public:
  explicit LocalUserKernelCreateContext(const KernelConf& kernel_conf)
      : user_op_conf_(kernel_conf.op_attribute().op_conf()) {}

  const user_op::UserOpConfWrapper& user_op_conf() const override { return user_op_conf_; }

 private:
  user_op::UserOpConfWrapper user_op_conf_;
};

class LocalUserKernelInitContext final : public user_op::KernelInitContext {
 public:
  explicit LocalUserKernelInitContext(DeviceCtx* device_ctx, const KernelConf& kernel_conf,
                                      const std::shared_ptr<const JobDesc> job_desc,
                                      TensorIndexMap bn_in_op2bn_index2input_tensor_index,
                                      TensorIndexMap bn_in_op2bn_index2output_tensor_index)
      : user_op::KernelInitContext(
          user_op::UserOpConfWrapper(kernel_conf.op_attribute().op_conf())),
        device_ctx_(device_ctx),
        base_ctx_(LocalUserKernelBaseContext(kernel_conf, job_desc,
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

 private:
  DeviceCtx* device_ctx_;
  LocalUserKernelBaseContext base_ctx_;
};

class LocalUserOpInferContext : public user_op::InferContext {
 public:
  LocalUserOpInferContext(const KernelConf& kernel_conf,
                          const std::shared_ptr<const JobDesc> job_desc,
                          TensorIndexMap bn_in_op2bn_index2input_tensor_index,
                          TensorIndexMap bn_in_op2bn_index2output_tensor_index)
      : user_op_conf_(kernel_conf.op_attribute().op_conf()),
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
  ~LocalUserOpInferContext() override = default;

  const user_op::TensorDesc* LogicalTensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                               int32_t index) const override {
    UNIMPLEMENTED();
  }
  user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                  int32_t index) override {
    auto it = arg2tensor_desc_.find(std::make_pair(arg_name, index));
    CHECK(it != arg2tensor_desc_.end()) << "Arg (" << arg_name << "," << index << ") is not found";
    // return it->second.get();
    return nullptr;
  }
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

  void Update(TensorsPtr inputs, TensorsPtr outputs) {
    input_tensors_ = inputs;
    output_tensors_ = outputs;

    arg2tensor_desc_.clear();
    auto UpdateArg2TensorDesc = [this](TensorsPtr tensors_ptr, TensorIndexMap tensor_index_map) {
      for (auto& pair : *tensor_index_map) {
        const auto& bn_in_op = pair.first;
        for (int64_t bn_index = 0; bn_index < pair.second.size(); bn_index++) {
          const auto tensor_index = pair.second[bn_index];
          // TODO: update arg2tensor_desc_ with real value
          // arg2tensor_desc[{bn_in_op, bn_index}] = std::make_shared<ZeroCopyTensorDesc>();
        }
      }
    };

    UpdateArg2TensorDesc(input_tensors_, bn_in_op2bn_index2input_tensor_index_);
    UpdateArg2TensorDesc(output_tensors_, bn_in_op2bn_index2output_tensor_index_);
  }

 private:
  user_op::UserOpConfWrapper user_op_conf_;
  std::shared_ptr<const JobDesc> job_desc_;
  ArgVec inputs_;
  ArgVec outputs_;
  TensorIndexMap bn_in_op2bn_index2input_tensor_index_;
  TensorIndexMap bn_in_op2bn_index2output_tensor_index_;
  TensorsPtr input_tensors_;
  TensorsPtr output_tensors_;
  HashMap<std::pair<std::string, int64_t>, std::shared_ptr<ZeroCopyTensorDesc>> arg2tensor_desc_;
};

class LocalUserKernelComputeContext final : public user_op::KernelComputeContext {
 public:
  explicit LocalUserKernelComputeContext(DeviceCtx* device_ctx, const KernelConf& kernel_conf,
                                         const std::shared_ptr<const JobDesc> job_desc,
                                         TensorIndexMap bn_in_op2bn_index2input_tensor_index,
                                         TensorIndexMap bn_in_op2bn_index2output_tensor_index)
      : user_op::KernelComputeContext(
          user_op::UserOpConfWrapper(kernel_conf.op_attribute().op_conf())),
        device_ctx_(device_ctx),
        base_ctx_(LocalUserKernelBaseContext(kernel_conf, job_desc,
                                             bn_in_op2bn_index2input_tensor_index,
                                             bn_in_op2bn_index2output_tensor_index)) {}
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
  const ParallelContext& parallel_ctx() const override { UNIMPLEMENTED(); }
  const JobDesc& job_desc() const override { return base_ctx_.job_desc(); }

  const ArgVec& inputs() const override { return base_ctx_.inputs(); }
  const ArgVec& outputs() const override { return base_ctx_.outputs(); }

  void Update(TensorsPtr inputs, TensorsPtr outputs, DeviceCtx* device_ctx) {
    input_tensors_ = inputs;
    output_tensors_ = outputs;
    device_ctx_ = device_ctx;
    base_ctx_.Update(inputs, outputs);
  }

 private:
  DeviceCtx* device_ctx_;
  LocalUserKernelBaseContext base_ctx_;
  TensorIndexMap bn_in_op2bn_index2input_tensor_index_;
  TensorIndexMap bn_in_op2bn_index2output_tensor_index_;
  TensorsPtr input_tensors_;
  TensorsPtr output_tensors_;
};

StatefulOpKernel::StatefulOpKernel(const std::shared_ptr<const JobDesc> job_desc,
                                   const KernelConf& kernel_conf,
                                   TensorIndexMap bn_in_op2bn_index2input_tensor_index,
                                   TensorIndexMap bn_in_op2bn_index2output_tensor_index)
    : job_desc_(job_desc),
      kernel_conf_(kernel_conf),
      bn_in_op2bn_index2input_tensor_index_(bn_in_op2bn_index2input_tensor_index),
      bn_in_op2bn_index2output_tensor_index_(bn_in_op2bn_index2output_tensor_index) {
  op_infer_ctx_.reset(new LocalUserOpInferContext(kernel_conf, job_desc,
                                                  bn_in_op2bn_index2input_tensor_index,
                                                  bn_in_op2bn_index2output_tensor_index));
  compute_ctx_.reset(new LocalUserKernelComputeContext(nullptr, kernel_conf, job_desc,
                                                       bn_in_op2bn_index2input_tensor_index,
                                                       bn_in_op2bn_index2output_tensor_index));
  init_ctx_.reset(new LocalUserKernelInitContext(nullptr, kernel_conf, job_desc,
                                                 bn_in_op2bn_index2input_tensor_index,
                                                 bn_in_op2bn_index2output_tensor_index));
  create_ctx_.reset(new LocalUserKernelCreateContext(kernel_conf));
  reg_ctx_.reset(new LocalUserKernelRegContext(kernel_conf, job_desc, bn_in_op2bn_index2input_tensor_index_,
                                bn_in_op2bn_index2output_tensor_index_));
  const auto* op_reg_val = user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(
      kernel_conf.op_attribute().op_conf().user_conf().op_type_name());
  CHECK_NOTNULL(op_reg_val);
  if (op_reg_val->physical_tensor_desc_infer_fn) {
    tensor_desc_infer_fn_ = op_reg_val->physical_tensor_desc_infer_fn;
  } else {
    UNIMPLEMENTED();
  }
  InitOpKernel(kernel_conf);
}

void StatefulOpKernel::InitOpKernel(const KernelConf& kernel_conf) {
  const std::string& op_type_name = kernel_conf.op_attribute().op_conf().user_conf().op_type_name();
  auto kernel_reg_val = CHECK_JUST(user_op::UserOpRegistryMgr::Get().GetOpKernelRegistryResult(
      op_type_name,
      *reg_ctx_));
  CHECK_NOTNULL(kernel_reg_val);
  kernel_.reset(kernel_reg_val->create_fn(create_ctx_.get()));
}

Maybe<void> StatefulOpKernel::TryInitOpKernelState(DeviceCtx* device_ctx) {
  if (state_ == nullptr) {
    init_ctx_->set_device_ctx(device_ctx);
    state_ = kernel_->CreateOpKernelState(init_ctx_.get());
  }
  return Maybe<void>::Ok();
}

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
