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
#include "oneflow/user/kernels/stateful_local_opkernel.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/attr_value_accessor.h"
#include "oneflow/core/framework/attr_value_map.h"

namespace oneflow {
namespace one {

template<class T>
class InputAndOutputListScope {
 public:
  InputAndOutputListScope(T* ctx, EagerBlobObjectList inputs, EagerBlobObjectList outputs) {
    ctx_ = ctx;
    ctx_->Update(inputs, outputs);
  }
  ~InputAndOutputListScope() { ctx_->Update(nullptr, nullptr); }

 private:
  T* ctx_;
};

int32_t TryGetTensorTupleIndex(
    const std::map<std::string, std::vector<int32_t>>& arg_name2bn_index2tensor_tuple_index_,
    const std::string& arg_name, const int32_t arg_index) {
  auto it = arg_name2bn_index2tensor_tuple_index_.find(arg_name);
  if (it != arg_name2bn_index2tensor_tuple_index_.end()) { return it->second.at(arg_index); }
  return -1;
};

void InitArgName2BnIndex2TensorTupleIndex(
    const ArgVec* indexed_arg_pairs,
    std::map<std::string, std::vector<int32_t>>* arg_name2bn_index2tensor_tuple_index) {
  for (int i = 0; i < indexed_arg_pairs->size(); i++) {
    const auto& pair = indexed_arg_pairs->at(i);
    const std::string& arg_name = pair.first;
    const int32_t bn_index = pair.second;
    // vector is auto created by [] if arg_name doesn't exist in map
    auto& bn_index2tensor_tuple_index = (*arg_name2bn_index2tensor_tuple_index)[arg_name];
    CHECK_EQ(bn_index2tensor_tuple_index.size(), bn_index);
    bn_index2tensor_tuple_index.push_back(i);
  }
};

ZeroCopyBaseContext::ZeroCopyBaseContext(const ArgVec* indexed_input_pairs,
                                         const ArgVec* indexed_output_pairs)
    : indexed_input_pairs_(indexed_input_pairs), indexed_output_pairs_(indexed_output_pairs) {
  InitArgName2BnIndex2TensorTupleIndex(indexed_input_pairs,
                                       &arg_name2bn_index2input_tensor_tuple_index_);
  InitArgName2BnIndex2TensorTupleIndex(indexed_output_pairs,
                                       &arg_name2bn_index2output_tensor_tuple_index_);
  for (int i = 0; i < indexed_input_pairs->size(); i++) {
    input_tensor_views_.push_back(std::make_unique<EagerBlobObjectTensorView>(
        [this, i]() -> eager::EagerBlobObject* { return input_tensors_->at(i).get(); }));
    input_tensor_desc_views_.push_back(std::make_unique<EagerBlobObjectTensorDescView>(
        [this, i]() -> eager::EagerBlobObject* { return input_tensors_->at(i).get(); }));
  }
  for (int i = 0; i < indexed_output_pairs->size(); i++) {
    output_tensor_views_.push_back(std::make_unique<EagerBlobObjectTensorView>(
        [this, i]() -> eager::EagerBlobObject* { return output_tensors_->at(i).get(); }));
    output_tensor_desc_views_.push_back(std::make_unique<EagerBlobObjectTensorDescView>(
        [this, i]() -> eager::EagerBlobObject* { return output_tensors_->at(i).get(); }));
  }
}

void ZeroCopyBaseContext::Update(EagerBlobObjectList inputs, EagerBlobObjectList outputs) {
  input_tensors_ = inputs;
  output_tensors_ = outputs;
}

user_op::TensorDesc* ZeroCopyBaseContext::TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                                     const int32_t index) const {
  int32_t i = TryGetTensorTupleIndex(arg_name2bn_index2input_tensor_tuple_index_, arg_name, index);
  if (i >= 0) { return input_tensor_desc_views_.at(i).get(); }
  i = TryGetTensorTupleIndex(arg_name2bn_index2output_tensor_tuple_index_, arg_name, index);
  if (i >= 0) { return output_tensor_desc_views_.at(i).get(); }
  LOG(FATAL) << "Arg (" << arg_name << "," << index << ") is not found";
  return nullptr;
}

user_op::Tensor* ZeroCopyBaseContext::Tensor4ArgNameAndIndex(const std::string& arg_name,
                                                             const int32_t index) const {
  int32_t i = TryGetTensorTupleIndex(arg_name2bn_index2input_tensor_tuple_index_, arg_name, index);
  if (i >= 0) { return input_tensor_views_.at(i).get(); }
  i = TryGetTensorTupleIndex(arg_name2bn_index2output_tensor_tuple_index_, arg_name, index);
  if (i >= 0) { return output_tensor_views_.at(i).get(); }
  LOG(FATAL) << "Arg (" << arg_name << "," << index << ") is not found";
  return nullptr;
}

LocalUserKernelBaseContext::LocalUserKernelBaseContext(const std::string& device_tag,
                                                       const ArgVec* indexed_input_pairs,
                                                       const ArgVec* indexed_output_pairs)
    : ZeroCopyBaseContext(indexed_input_pairs, indexed_output_pairs),
      device_tag_(device_tag),
      device_type_(CHECK_JUST(DeviceType4DeviceTag(device_tag_))) {}

class LocalUserKernelRegContext final : public user_op::KernelRegContext {
 public:
  explicit LocalUserKernelRegContext(const std::string& device_tag,
                                     const user_op::UserOpConfWrapper* user_op_conf,
                                     const ArgVec* index_input_pairs,
                                     const ArgVec* indexed_output_pairs)
      : user_op_conf_(user_op_conf),
        base_ctx_(device_tag, index_input_pairs, indexed_output_pairs) {}
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

  void Update(EagerBlobObjectList inputs, EagerBlobObjectList outputs) {
    base_ctx_.Update(inputs, outputs);
  }

  const user_op::UserOpConfWrapper& user_op_conf() const override { return *user_op_conf_; }

 private:
  const user_op::UserOpConfWrapper* user_op_conf_;
  LocalUserKernelBaseContext base_ctx_;
};

class LocalUserKernelCreateContext final : public user_op::KernelCreateContext {
 public:
  explicit LocalUserKernelCreateContext(const user_op::UserOpConfWrapper* user_op_conf)
      : user_op_conf_(user_op_conf) {}

 private:
  const user_op::UserOpConfWrapper& user_op_conf() const override { return *user_op_conf_; }
  const std::shared_ptr<user_op::AttrVal>& Attr4AttrName(
      const std::string& attr_name) const override {
    return user_op_conf().Attr4AttrName(attr_name);
  }

  const user_op::UserOpConfWrapper* user_op_conf_;
};

class LocalUserKernelInitContext final : public user_op::KernelInitContext {
 public:
  explicit LocalUserKernelInitContext(DeviceCtx* device_ctx, const std::string& device_tag,
                                      const user_op::UserOpConfWrapper* user_op_conf,
                                      const ArgVec* index_input_pairs,
                                      const ArgVec* indexed_output_pairs,
                                      EagerBlobObjectList inputs, EagerBlobObjectList outputs)
      : user_op_conf_(user_op_conf),
        device_ctx_(device_ctx),
        base_ctx_(device_tag, index_input_pairs, indexed_output_pairs) {
    base_ctx_.Update(inputs, outputs);
  }
  ~LocalUserKernelInitContext() override = default;

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
  const std::shared_ptr<user_op::AttrVal>& Attr4AttrName(
      const std::string& attr_name) const override {
    return user_op_conf().Attr4AttrName(attr_name);
  }

  const user_op::UserOpConfWrapper& user_op_conf() const override { return *user_op_conf_; }

  const user_op::UserOpConfWrapper* user_op_conf_;
  DeviceCtx* device_ctx_;
  LocalUserKernelBaseContext base_ctx_;
};

LocalUserOpInferContext::LocalUserOpInferContext(const user_op::UserOpConfWrapper* user_op_conf,
                                                 const ArgVec* index_input_pairs,
                                                 const ArgVec* indexed_output_pairs)
    : user_op_conf_(user_op_conf), zero_copy_base_ctx_(index_input_pairs, indexed_output_pairs) {}

user_op::TensorDesc* LocalUserOpInferContext::TensorDesc4ArgNameAndIndex(
    const std::string& arg_name, int32_t index) {
  return zero_copy_base_ctx_.TensorDesc4ArgNameAndIndex(arg_name, index);
}

void LocalUserOpInferContext::Update(EagerBlobObjectList inputs, EagerBlobObjectList outputs) {
  zero_copy_base_ctx_.Update(inputs, outputs);
}

LocalUserKernelComputeContext::LocalUserKernelComputeContext(
    DeviceCtx* device_ctx, const std::string& device_tag,
    const user_op::UserOpConfWrapper* user_op_conf, const ArgVec* index_input_pairs,
    const ArgVec* indexed_output_pairs)
    : user_op_conf_(user_op_conf),
      device_ctx_(device_ctx),
      base_ctx_(device_tag, index_input_pairs, indexed_output_pairs) {}

void LocalUserKernelComputeContext::Update(EagerBlobObjectList inputs, EagerBlobObjectList outputs,
                                           DeviceCtx* device_ctx) {
  device_ctx_ = device_ctx;
  base_ctx_.Update(inputs, outputs);
}

Maybe<void> InitTensorTupleIndexes4Bns(const std::shared_ptr<const OperatorConf>& op_conf,
                                       const ArgVec* indexed_input_pairs,
                                       const ArgVec* indexed_output_pairs,
                                       std::vector<int64_t>* input_tuple_indexes4const_ibns,
                                       std::vector<int64_t>* input_tuple_indexes4mut_ibns,
                                       std::vector<int64_t>* output_tuple_indexes4mut_obns,
                                       std::vector<int64_t>* output_tuple_indexes4mut2_obns) {
  const auto* op_reg_val =
      user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_conf->user_conf().op_type_name());
  CHECK_NOTNULL_OR_RETURN(op_reg_val);

  ArgModifierSignature arg_modifier_signature;
  for (const auto& pair : *indexed_input_pairs) {
    const std::string ibn = GenRepeatedBn(pair.first, pair.second);
    arg_modifier_signature.mutable_ibn2input_blob_modifier()->insert(
        {ibn, user_op::InputArgModifier()});
  }
  for (const auto& pair : *indexed_output_pairs) {
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
    op_reg_val->input_arg_modify_fn(GetInputArgModifierFn, op_conf_wrapper);
  }
  if (op_reg_val->output_arg_modify_fn) {
    user_op::GetOutputArgModifier GetOutputArgModifierFn =
        [&arg_modifier_signature](const std::string& in_arg_name,
                                  int32_t in_arg_index) -> user_op::OutputArgModifier* {
      const std::string obn = GenRepeatedBn(in_arg_name, in_arg_index);
      auto* map = arg_modifier_signature.mutable_obn2output_blob_modifier();
      return &map->at(obn);
    };
    op_reg_val->output_arg_modify_fn(GetOutputArgModifierFn, op_conf_wrapper);
  }

  for (int i = 0; i < indexed_input_pairs->size(); i++) {
    const auto& pair = indexed_input_pairs->at(i);
    const std::string ibn = GenRepeatedBn(pair.first, pair.second);
    if (arg_modifier_signature.ibn2input_blob_modifier().at(ibn).is_mutable()) {
      input_tuple_indexes4mut_ibns->push_back(i);
    } else {
      input_tuple_indexes4const_ibns->push_back(i);
    }
  }

  for (int i = 0; i < indexed_output_pairs->size(); i++) {
    const auto& pair = indexed_output_pairs->at(i);
    const std::string obn = GenRepeatedBn(pair.first, pair.second);
    if (arg_modifier_signature.obn2output_blob_modifier().at(obn).header_infered_before_compute()) {
      output_tuple_indexes4mut_obns->push_back(i);
    } else {
      output_tuple_indexes4mut2_obns->push_back(i);
    }
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<StatefulOpKernel> StatefulOpKernel::New(
    const std::shared_ptr<OperatorConf>& op_conf, const std::shared_ptr<MemoryCase>& mem_case,
    const std::shared_ptr<const ParallelDesc>& parallel_desc,
    const std::shared_ptr<ArgVec> indexed_input_pairs,
    const std::shared_ptr<ArgVec> indexed_output_pairs) {
  auto opkernel = std::shared_ptr<StatefulOpKernel>(new StatefulOpKernel());
  opkernel->op_conf_ = op_conf;
  opkernel->user_op_conf_.reset(new user_op::UserOpConfWrapper(op_conf));
  opkernel->mem_case_ = mem_case;
  opkernel->indexed_input_pairs_ = indexed_input_pairs;
  opkernel->indexed_output_pairs_ = indexed_output_pairs;
  opkernel->need_check_mem_case_ = true;

  const std::string& device_tag = op_conf->device_tag();
  opkernel->op_infer_ctx_.reset(new LocalUserOpInferContext(
      opkernel->user_op_conf_.get(), indexed_input_pairs.get(), indexed_output_pairs.get()));
  opkernel->compute_ctx_.reset(
      new LocalUserKernelComputeContext(nullptr, device_tag, opkernel->user_op_conf_.get(),
                                        indexed_input_pairs.get(), indexed_output_pairs.get()));
  opkernel->create_ctx_.reset(new LocalUserKernelCreateContext(opkernel->user_op_conf_.get()));
  opkernel->reg_ctx_.reset(new LocalUserKernelRegContext(device_tag, opkernel->user_op_conf_.get(),
                                                         indexed_input_pairs.get(),
                                                         indexed_output_pairs.get()));
  const auto* op_reg_val =
      user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_conf->user_conf().op_type_name());
  CHECK_NOTNULL_OR_RETURN(op_reg_val);
  if (op_reg_val->physical_tensor_desc_infer_fn) {
    opkernel->tensor_desc_infer_fn_ = op_reg_val->physical_tensor_desc_infer_fn;
  } else {
    return Error::Unimplemented();
  }
  opkernel->data_type_infer_fn_ = op_reg_val->data_type_infer_fn;

  JUST(InitTensorTupleIndexes4Bns(
      op_conf, indexed_input_pairs.get(), indexed_output_pairs.get(),
      &opkernel->input_tuple_indexes4const_ibns_, &opkernel->input_tuple_indexes4mut_ibns_,
      &opkernel->output_tuple_indexes4mut_obns_, &opkernel->output_tuple_indexes4mut2_obns_));

  opkernel->tmp_blob_object_.reset(
      new eager::EagerBlobObject(opkernel->mem_case_, std::make_shared<Shape>(), DataType::kChar,
                                 std::make_shared<eager::TensorBuffer>()));
  opkernel->infer_local_dep_object_ = std::make_shared<VmLocalDepObject>(parallel_desc);
  opkernel->compute_local_dep_object_ = std::make_shared<VmLocalDepObject>(parallel_desc);
  return opkernel;
}

StatefulOpKernel::~StatefulOpKernel() = default;

Maybe<const user_op::OpKernel*> StatefulOpKernel::ChooseOpKernel(EagerBlobObjectList inputs,
                                                                 EagerBlobObjectList outputs) {
  InputAndOutputListScope<LocalUserKernelRegContext> reg_ctx_scope(reg_ctx_.get(), inputs, outputs);
  const auto& op_type_name = user_op_conf_->op_type_name();
  const auto* kernel_reg_val =
      JUST(user_op::UserOpRegistryMgr::Get().GetOpKernelRegistryResult(op_type_name, *reg_ctx_));
  CHECK_NOTNULL(kernel_reg_val);
  // find cached kernel by registry result
  auto it = op_kernel_map_.find(kernel_reg_val);
  if (it != op_kernel_map_.end()) { return it->second.get(); }

  auto* kernel = kernel_reg_val->create_fn(create_ctx_.get());
  op_kernel_map_.emplace(kernel_reg_val, std::shared_ptr<const user_op::OpKernel>(kernel));

  infer_tmp_size_fn_map_.emplace(kernel, &kernel_reg_val->infer_tmp_size_fn);

  return kernel;
}

void StatefulOpKernel::TryInitOpKernelState(const user_op::OpKernel* op_kernel,
                                            DeviceCtx* device_ctx, EagerBlobObjectList inputs,
                                            EagerBlobObjectList outputs,
                                            user_op::OpKernelState** state) {
  auto it = op_kernel_state_map_.find(op_kernel);
  if (it != op_kernel_state_map_.end()) {
    *state = it->second.get();
    return;
  }

  auto init_ctx = std::make_shared<LocalUserKernelInitContext>(
      device_ctx, op_conf_->device_tag(), user_op_conf_.get(), indexed_input_pairs_.get(),
      indexed_output_pairs_.get(), inputs, outputs);
  auto created_state = op_kernel->CreateOpKernelState(init_ctx.get());
  op_kernel_state_map_.emplace(op_kernel, created_state);
  *state = created_state.get();
}

const user_op::InferTmpSizeFn& StatefulOpKernel::GetInferTmpSizeFn(
    const user_op::OpKernel* op_kernel) const {
  return *infer_tmp_size_fn_map_.at(op_kernel);
}

eager::EagerBlobObject* StatefulOpKernel::mut_temp_blob_object() { return tmp_blob_object_.get(); }

user_op::TensorDescInferFn StatefulOpKernel::TensorDescInferFn() const {
  return tensor_desc_infer_fn_;
}

user_op::DataTypeInferFn StatefulOpKernel::DataTypeInferFn() const { return data_type_infer_fn_; }

LocalUserOpInferContext* StatefulOpKernel::UpdateInferContext(EagerBlobObjectList inputs,
                                                              EagerBlobObjectList outputs) {
  op_infer_ctx_->Update(inputs, outputs);
  return op_infer_ctx_.get();
}

LocalUserKernelComputeContext* StatefulOpKernel::UpdateComputeContext(EagerBlobObjectList inputs,
                                                                      EagerBlobObjectList outputs,
                                                                      DeviceCtx* device_ctx) {
  compute_ctx_->Update(inputs, outputs, device_ctx);
  return compute_ctx_.get();
}

void StatefulOpKernel::UpdateOpAttrs(const AttrValueMap& attrs) {
  auto* user_op_conf = op_conf_->mutable_user_conf();
  for (const auto& it : attrs) {
    AttrValue attr_val;
    it.second->ToProto(&attr_val);
    (*(user_op_conf->mutable_attr()))[it.first] = attr_val;
  }
  *user_op_conf_ = user_op::UserOpConfWrapper(op_conf_);
}
}  // namespace one
}  // namespace oneflow
