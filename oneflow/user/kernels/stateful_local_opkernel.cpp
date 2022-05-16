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
#include "oneflow/core/framework/attr_value_accessor.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/framework/consistent_tensor_infer_cache.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/profiler/profiler.h"

namespace oneflow {
namespace one {

int32_t TryGetTensorTupleIndex(const std::unordered_map<std::string, std::vector<int32_t>>&
                                   arg_name2bn_index2tensor_tuple_index,
                               const std::string& arg_name, const int32_t arg_index) {
  auto it = arg_name2bn_index2tensor_tuple_index.find(arg_name);
  if (it != arg_name2bn_index2tensor_tuple_index.end()) { return it->second.at(arg_index); }
  return -1;
}

ZeroCopyBaseContext::ZeroCopyBaseContext(const std::shared_ptr<const ArgTuple>& input_arg_tuple,
                                         const std::shared_ptr<const ArgTuple>& output_arg_tuple)
    : ZeroCopyBaseContext(input_arg_tuple, output_arg_tuple, nullptr) {}

ZeroCopyBaseContext::ZeroCopyBaseContext(const std::shared_ptr<const ArgTuple>& input_arg_tuple,
                                         const std::shared_ptr<const ArgTuple>& output_arg_tuple,
                                         vm::EagerBlobObject* tmp_buffer)
    : input_arg_tuple_(input_arg_tuple), output_arg_tuple_(output_arg_tuple) {
  for (int i = 0; i < input_arg_tuple->size(); i++) {
    input_tensor_views_.emplace_back(std::make_unique<EagerBlobObjectTensorView>(
        [this, i]() -> vm::EagerBlobObject* { return input_tensors_->at(i).get(); }));
    input_tensor_desc_views_.emplace_back(std::make_unique<EagerBlobObjectTensorDescView>(
        [this, i]() -> vm::EagerBlobObject* { return input_tensors_->at(i).get(); }));
    input_consistent_tensor_meta_views_.emplace_back(
        std::make_unique<ConsistentTensorMetaTensorDescView>(
            [this, i]() -> Symbol<ConsistentTensorMeta> {
              return CHECK_NOTNULL(consistent_tensor_infer_result_)->input_tensor_metas().at(i);
            }));
  }
  for (int i = 0; i < output_arg_tuple->size(); i++) {
    output_tensor_views_.emplace_back(std::make_unique<EagerBlobObjectTensorView>(
        [this, i]() -> vm::EagerBlobObject* { return output_tensors_->at(i).get(); }));
    output_tensor_desc_views_.emplace_back(std::make_unique<EagerBlobObjectTensorDescView>(
        [this, i]() -> vm::EagerBlobObject* { return output_tensors_->at(i).get(); }));
    output_consistent_tensor_meta_views_.emplace_back(
        std::make_unique<ConsistentTensorMetaTensorDescView>(
            [this, i]() -> Symbol<ConsistentTensorMeta> {
              return CHECK_NOTNULL(consistent_tensor_infer_result_)->output_tensor_metas().at(i);
            }));
  }
  if (tmp_buffer != nullptr) {
    tmp_buffer_view_.reset(new EagerBlobObjectTensorView([tmp_buffer]() { return tmp_buffer; }));
  }
}

void ZeroCopyBaseContext::Update(EagerBlobObjectListRawPtr inputs,
                                 EagerBlobObjectListRawPtr outputs,
                                 ConsistentTensorInferResultRawPtr consistent_tensor_infer_result) {
  input_tensors_ = inputs;
  output_tensors_ = outputs;
  consistent_tensor_infer_result_ = consistent_tensor_infer_result;
}

Optional<Symbol<ParallelDesc>> ZeroCopyBaseContext::parallel_desc() const {
  if (!consistent_tensor_infer_result_) { return Optional<Symbol<ParallelDesc>>(); }
  if (!consistent_tensor_infer_result_->input_tensor_metas().empty()) {
    return consistent_tensor_infer_result_->input_tensor_metas().at(0)->parallel_desc();
  } else if (!consistent_tensor_infer_result_->output_tensor_metas().empty()) {
    return consistent_tensor_infer_result_->output_tensor_metas().at(0)->parallel_desc();
  } else {
    UNIMPLEMENTED();
    return Optional<Symbol<ParallelDesc>>();
  }
}

namespace {
ParallelContext MakeSingleDeviceParallelCtx() {
  ParallelContext single_device_parallel_ctx;
  single_device_parallel_ctx.set_parallel_id(0);
  single_device_parallel_ctx.set_parallel_num(1);
  return single_device_parallel_ctx;
}
}  // namespace

const ParallelContext& ZeroCopyBaseContext::parallel_ctx() const {
  const auto& parallel_desc = this->parallel_desc();
  if (parallel_desc.has_value()) {
    const auto& parallel_desc_symbol = CHECK_JUST(parallel_desc);
    return *CHECK_JUST(GetParallelContext4CurrentProcessCtx(parallel_desc_symbol));
  } else {
    static ParallelContext single_device_parallel_ctx(MakeSingleDeviceParallelCtx());
    return single_device_parallel_ctx;
  }
}

#define RETURN_IF_FOUND(inputs, outputs, post_action)                                             \
  int32_t i = TryGetTensorTupleIndex(input_arg_tuple_->arg_name2bn_index2tensor_tuple_index(),    \
                                     arg_name, index);                                            \
  if (i >= 0) { return (inputs).at(i) post_action; }                                              \
  i = TryGetTensorTupleIndex(output_arg_tuple_->arg_name2bn_index2tensor_tuple_index(), arg_name, \
                             index);                                                              \
  if (i >= 0) { return (outputs).at(i) post_action; }

user_op::TensorDesc* ZeroCopyBaseContext::TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                                     const int32_t index) const {
  RETURN_IF_FOUND(input_tensor_desc_views_, output_tensor_desc_views_, .get());
  return nullptr;
}

user_op::Tensor* ZeroCopyBaseContext::Tensor4ArgNameAndIndex(const std::string& arg_name,
                                                             const int32_t index) const {
  RETURN_IF_FOUND(input_tensor_views_, output_tensor_views_, .get());
  if (arg_name == "tmp_buffer" && index == 0) { return CHECK_NOTNULL(tmp_buffer_view_.get()); }
  return nullptr;
}

const ConsistentTensorMeta* ZeroCopyBaseContext::ConsistentTensorMeta4ArgNameAndIndex(
    const std::string& arg_name, const int32_t index) const {
  RETURN_IF_FOUND(consistent_tensor_infer_result_->input_tensor_metas(),
                  consistent_tensor_infer_result_->output_tensor_metas(),
                  .shared_from_symbol().get());
  return nullptr;
}

const ConsistentTensorMetaTensorDescView*
ZeroCopyBaseContext::ConsistentTensorMetaView4ArgNameAndIndex(const std::string& arg_name,
                                                              const int32_t index) const {
  RETURN_IF_FOUND(input_consistent_tensor_meta_views_, output_consistent_tensor_meta_views_,
                  .get());
  return nullptr;
}

LocalUserKernelBaseContext::LocalUserKernelBaseContext(
    const std::string& device_tag, const std::shared_ptr<const ArgTuple>& input_arg_tuple,
    const std::shared_ptr<const ArgTuple>& output_arg_tuple)
    : LocalUserKernelBaseContext(device_tag, input_arg_tuple, output_arg_tuple, nullptr) {}

LocalUserKernelBaseContext::LocalUserKernelBaseContext(
    const std::string& device_tag, const std::shared_ptr<const ArgTuple>& input_arg_tuple,
    const std::shared_ptr<const ArgTuple>& output_arg_tuple, vm::EagerBlobObject* tmp_buffer)
    : ZeroCopyBaseContext(input_arg_tuple, output_arg_tuple, tmp_buffer),
      device_tag_(device_tag),
      device_type_(CHECK_JUST(DeviceType4DeviceTag(device_tag_))),
      tmp_buffer_(tmp_buffer) {}

class LocalUserKernelRegContext final : public user_op::KernelRegContext {
 public:
  explicit LocalUserKernelRegContext(const std::string& device_tag,
                                     const user_op::UserOpConfWrapper* user_op_conf,
                                     ComposedAttrMap* composed_attrs,
                                     const std::shared_ptr<const ArgTuple>& input_arg_tuple,
                                     const std::shared_ptr<const ArgTuple>& output_arg_tuple)
      : user_op_conf_(user_op_conf),
        composed_attrs_(composed_attrs),
        base_ctx_(device_tag, input_arg_tuple, output_arg_tuple) {}
  ~LocalUserKernelRegContext() = default;

  DeviceType device_type() const override { return base_ctx_.device_type(); }
  const std::string& device_tag() const override { return base_ctx_.device_tag(); }
  const ParallelContext& parallel_ctx() const override { return base_ctx_.parallel_ctx(); }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return base_ctx_.TensorDesc4ArgNameAndIndex(arg_name, index);
  }
  const ArgVec& inputs() const override { return base_ctx_.inputs(); }
  const ArgVec& outputs() const override { return base_ctx_.outputs(); }

  void Update(const AttrMap& attrs, EagerBlobObjectListRawPtr inputs,
              EagerBlobObjectListRawPtr outputs,
              ConsistentTensorInferResultRawPtr consistent_tensor_infer_result) {
    composed_attrs_->ResetPrior(attrs);
    base_ctx_.Update(inputs, outputs, consistent_tensor_infer_result);
  }

  const user_op::UserOpConfWrapper& user_op_conf() const override { return *user_op_conf_; }

 private:
  const user_op::UserOpConfWrapper* user_op_conf_;
  ComposedAttrMap* composed_attrs_;
  LocalUserKernelBaseContext base_ctx_;

  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return composed_attrs_->Attr4Name(attr_name);
  }
};

class LocalUserKernelInitAndCacheContext final : public user_op::KernelInitContext,
                                                 public user_op::KernelCacheContext {
 public:
  explicit LocalUserKernelInitAndCacheContext(
      DeviceCtx* device_ctx, const std::string& device_tag,
      const user_op::UserOpConfWrapper* user_op_conf,
      const std::shared_ptr<const ArgTuple>& input_arg_tuple,
      const std::shared_ptr<const ArgTuple>& output_arg_tuple, EagerBlobObjectListRawPtr inputs,
      EagerBlobObjectListRawPtr outputs,
      ConsistentTensorInferResultRawPtr consistent_tensor_infer_result,
      const ComposedAttrMap* composed_attrs)
      : user_op_conf_(user_op_conf),
        device_ctx_(device_ctx),
        base_ctx_(device_tag, input_arg_tuple, output_arg_tuple),
        composed_attrs_(composed_attrs) {
    base_ctx_.Update(inputs, outputs, consistent_tensor_infer_result);
  }
  ~LocalUserKernelInitAndCacheContext() override = default;

  ep::Stream* stream() override {
    CHECK(device_ctx_);
    return device_ctx_->stream();
  }

  DeviceType device_type() const override { return base_ctx_.device_type(); }
  const ParallelContext& parallel_ctx() const override { return base_ctx_.parallel_ctx(); }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return base_ctx_.TensorDesc4ArgNameAndIndex(arg_name, index);
  }
  const user_op::TensorDesc* LogicalTensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                               int32_t index) const override {
    return base_ctx_.ConsistentTensorMetaView4ArgNameAndIndex(arg_name, index);
  }
  const SbpParallel& SbpParallel4ArgNameAndIndex(const std::string& arg_name,
                                                 int32_t index) const override {
    const auto& nd_sbp = NdSbp4ArgNameAndIndex(arg_name, index);
    CHECK_EQ(nd_sbp.sbp_parallel_size(), 1);
    return nd_sbp.sbp_parallel(0);
  }

  const NdSbp& NdSbp4ArgNameAndIndex(const std::string& arg_name, int32_t index) const override {
    return *CHECK_NOTNULL(base_ctx_.ConsistentTensorMeta4ArgNameAndIndex(arg_name, index))
                ->nd_sbp();
  }

  const ArgVec& inputs() const override { return base_ctx_.inputs(); }
  const ArgVec& outputs() const override { return base_ctx_.outputs(); }
  const ParallelDesc& parallel_desc() const override {
    return *CHECK_JUST(base_ctx_.parallel_desc());
  }

 private:
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return composed_attrs_->Attr4Name(attr_name);
  }

  const user_op::UserOpConfWrapper& user_op_conf() const override { return *user_op_conf_; }

  const user_op::UserOpConfWrapper* user_op_conf_;
  DeviceCtx* device_ctx_;
  LocalUserKernelBaseContext base_ctx_;
  const ComposedAttrMap* composed_attrs_;
};

LocalUserOpInferContext::LocalUserOpInferContext(
    const user_op::UserOpConfWrapper* user_op_conf, const ComposedAttrMap* composed_attrs,
    const std::shared_ptr<const ArgTuple>& input_arg_tuple,
    const std::shared_ptr<const ArgTuple>& output_arg_tuple)
    : user_op_conf_(user_op_conf),
      composed_attrs_(composed_attrs),
      zero_copy_base_ctx_(input_arg_tuple, output_arg_tuple) {}

user_op::TensorDesc* LocalUserOpInferContext::TensorDesc4ArgNameAndIndex(
    const std::string& arg_name, int32_t index) {
  return zero_copy_base_ctx_.TensorDesc4ArgNameAndIndex(arg_name, index);
}

void LocalUserOpInferContext::Update(
    EagerBlobObjectListRawPtr inputs, EagerBlobObjectListRawPtr outputs,
    ConsistentTensorInferResultRawPtr consistent_tensor_infer_result) {
  zero_copy_base_ctx_.Update(inputs, outputs, consistent_tensor_infer_result);
}

LocalUserKernelComputeContext::LocalUserKernelComputeContext(
    DeviceCtx* device_ctx, const std::string& device_tag,
    const user_op::UserOpConfWrapper* user_op_conf, const ComposedAttrMap* composed_attrs,
    const std::shared_ptr<const ArgTuple>& input_arg_tuple,
    const std::shared_ptr<const ArgTuple>& output_arg_tuple, vm::EagerBlobObject* tmp_buffer)
    : user_op_conf_(user_op_conf),
      composed_attrs_(composed_attrs),
      device_ctx_(device_ctx),
      base_ctx_(device_tag, input_arg_tuple, output_arg_tuple, tmp_buffer) {}

void LocalUserKernelComputeContext::Update(
    EagerBlobObjectListRawPtr inputs, EagerBlobObjectListRawPtr outputs,
    ConsistentTensorInferResultRawPtr consistent_tensor_infer_result, DeviceCtx* device_ctx) {
  device_ctx_ = device_ctx;
  base_ctx_.Update(inputs, outputs, consistent_tensor_infer_result);
}

Maybe<void> InitTensorTupleIndexes4Bns(const std::shared_ptr<const OperatorConf>& op_conf,
                                       const ArgVec& indexed_input_pairs,
                                       const ArgVec& indexed_output_pairs,
                                       std::vector<int64_t>* input_tuple_indexes4const_ibns,
                                       std::vector<int64_t>* input_tuple_indexes4mut_ibns,
                                       std::vector<int64_t>* output_tuple_indexes4mut_obns,
                                       std::vector<int64_t>* output_tuple_indexes4mut2_obns) {
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
    } else {
      output_tuple_indexes4mut2_obns->emplace_back(i);
    }
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<StatefulLocalOpKernel> StatefulLocalOpKernel::New(
    const std::shared_ptr<OperatorConf>& op_conf, const Symbol<Stream>& stream,
    const AttrMap& base_attrs, const std::shared_ptr<const ParallelDesc>& parallel_desc,
    const std::shared_ptr<const ArgTuple>& input_arg_tuple,
    const std::shared_ptr<const ArgTuple>& output_arg_tuple) {
  auto opkernel = std::shared_ptr<StatefulLocalOpKernel>(new StatefulLocalOpKernel());
  opkernel->op_conf_ = op_conf;
  opkernel->user_op_conf_.reset(new user_op::UserOpConfWrapper(op_conf));
  opkernel->stream_ = stream;
  opkernel->composed_attrs_for_scheduler_thread_.reset(new ComposedAttrMap(base_attrs));
  opkernel->composed_attrs_for_main_thread_.reset(new ComposedAttrMap(base_attrs));
  opkernel->input_arg_tuple_ = input_arg_tuple;
  opkernel->output_arg_tuple_ = output_arg_tuple;
  opkernel->need_check_mem_case_ = true;

  opkernel->tmp_blob_object_.reset(new vm::EagerBlobObject(
      opkernel->mem_case(), std::make_shared<Shape>(), std::make_shared<Stride>(), DataType::kChar,
      std::make_shared<vm::TensorStorage>()));

  const std::string& device_tag = op_conf->device_tag();
  const user_op::UserOpConfWrapper* user_op_conf = opkernel->user_op_conf_.get();
  opkernel->op_infer_ctx_for_scheduler_thread_.reset(new LocalUserOpInferContext(
      user_op_conf, opkernel->composed_attrs_for_scheduler_thread_.get(), input_arg_tuple,
      output_arg_tuple));
  opkernel->compute_ctx_.reset(new LocalUserKernelComputeContext(
      nullptr, device_tag, user_op_conf, opkernel->composed_attrs_for_scheduler_thread_.get(),
      input_arg_tuple, output_arg_tuple, opkernel->mut_temp_blob_object()));
  opkernel->reg_ctx_.reset(new LocalUserKernelRegContext(
      device_tag, user_op_conf, opkernel->composed_attrs_for_main_thread_.get(), input_arg_tuple,
      output_arg_tuple));
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
      &opkernel->output_tuple_indexes4mut2_obns_));

  return opkernel;
}

StatefulLocalOpKernel::~StatefulLocalOpKernel() = default;

Maybe<void> StatefulLocalOpKernel::ChooseOpKernel(
    const user_op::OpKernel** user_opkernel, bool* need_temp_storage, const AttrMap& attrs,
    EagerBlobObjectListRawPtr inputs, EagerBlobObjectListRawPtr outputs,
    ConsistentTensorInferResultRawPtr consistent_tensor_infer_result) {
  OF_PROFILER_RANGE_GUARD("ChooseOpKernel");
  reg_ctx_->Update(attrs, inputs, outputs, consistent_tensor_infer_result);

  DataType primary_dtype = kInvalidDataType;
  if (likely(!inputs->empty())) {
    primary_dtype = (*inputs)[0]->data_type();
  } else if (likely(!outputs->empty())) {
    primary_dtype = (*outputs)[0]->data_type();
  } else {
    // do nothing
  }

  for (const auto& pair : dtype2cached_kernels_[primary_dtype]) {
    if (likely(pair.first->is_matched_hob->get(*reg_ctx_))) {
      reg_ctx_->Update(AttrMap{}, nullptr, nullptr, nullptr);
      *need_temp_storage = pair.first->need_temp_storage;
      *user_opkernel = pair.second.get();
      return Maybe<void>::Ok();
    }
  }

  OF_PROFILER_RANGE_GUARD("fallback");

  const auto& op_type_name = user_op_conf_->op_type_name();
  const auto* kernel_reg_val =
      JUST(user_op::UserOpRegistryMgr::Get().GetOpKernelRegistryResult(op_type_name, *reg_ctx_));
  CHECK_NOTNULL(kernel_reg_val);
  auto* kernel = kernel_reg_val->create_fn();
  dtype2cached_kernels_[primary_dtype].push_back(
      {kernel_reg_val, std::shared_ptr<const user_op::OpKernel>(kernel)});

  infer_tmp_size_fn_map_.emplace(kernel, &kernel_reg_val->infer_tmp_size_fn);
  reg_ctx_->Update(AttrMap{}, nullptr, nullptr, nullptr);
  *need_temp_storage = kernel_reg_val->need_temp_storage;
  *user_opkernel = kernel;
  return Maybe<void>::Ok();
}

void StatefulLocalOpKernel::TryInitOpKernelStateAndCache(
    const user_op::OpKernel* op_kernel, DeviceCtx* device_ctx, EagerBlobObjectListRawPtr inputs,
    EagerBlobObjectListRawPtr outputs,
    ConsistentTensorInferResultRawPtr consistent_tensor_infer_result,
    user_op::OpKernelState** state, user_op::OpKernelCache** cache) {
  LocalUserKernelInitAndCacheContext init_and_cache_ctx(
      device_ctx, op_conf_->device_tag(), user_op_conf_.get(), input_arg_tuple_, output_arg_tuple_,
      inputs, outputs, consistent_tensor_infer_result, composed_attrs_for_scheduler_thread());
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

const user_op::InferTmpSizeFn& StatefulLocalOpKernel::GetInferTmpSizeFn(
    const user_op::OpKernel* op_kernel) const {
  return *infer_tmp_size_fn_map_.at(op_kernel);
}

vm::EagerBlobObject* StatefulLocalOpKernel::mut_temp_blob_object() {
  return tmp_blob_object_.get();
}

user_op::TensorDescInferFn StatefulLocalOpKernel::TensorDescInferFn() const {
  return tensor_desc_infer_fn_;
}

user_op::DataTypeInferFn StatefulLocalOpKernel::DataTypeInferFn() const {
  return data_type_infer_fn_;
}

LocalUserKernelComputeContext* StatefulLocalOpKernel::UpdateComputeContext(
    EagerBlobObjectListRawPtr inputs, EagerBlobObjectListRawPtr outputs,
    ConsistentTensorInferResultRawPtr consistent_tensor_infer_result, DeviceCtx* device_ctx) {
  compute_ctx_->Update(inputs, outputs, consistent_tensor_infer_result, device_ctx);
  return compute_ctx_.get();
}

}  // namespace one
}  // namespace oneflow
