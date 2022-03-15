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
#include "oneflow/core/framework/consistent_tensor_infer_cache.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/user_op_registry_manager.h"

namespace oneflow {
namespace one {

namespace {

bool OptionalEqual(const Optional<Symbol<NdSbp>>& lhs, const Optional<Symbol<NdSbp>>& rhs) {
  if (lhs.has_value() != rhs.has_value()) { return false; }
  if (!lhs.has_value()) { return true; }
  return CHECK_JUST(lhs) == CHECK_JUST(rhs);
}

}  // namespace

size_t InputConsistentTensorMeta::hash_value() const {
  size_t hash_value = std::hash<Symbol<ConsistentTensorMeta>>()(tensor_meta());
  if (consumer_nd_sbp_constraint().has_value()) {
    hash_value ^= std::hash<Symbol<NdSbp>>()(CHECK_JUST(consumer_nd_sbp_constraint()));
  }
  return hash_value;
}

bool InputConsistentTensorMeta::operator==(const InputConsistentTensorMeta& other) const {
  return this->tensor_meta() == other.tensor_meta()
         && OptionalEqual(this->consumer_nd_sbp_constraint(), other.consumer_nd_sbp_constraint());
}

void InputConsistentTensorMeta::assign(Symbol<ConsistentTensorMeta> tensor_meta,
                                       const Optional<Symbol<NdSbp>>& consumer_nd_sbp_constraint) {
  tensor_meta_ = tensor_meta;
  consumer_nd_sbp_constraint_ = consumer_nd_sbp_constraint;
}

size_t ConsistentTensorMetaInferArgs::hash_value() const {
  size_t hash_value = std::hash<AttrMap>()(attrs_);
  const auto& tensor_meta_hash_functor = std::hash<InputConsistentTensorMeta>();
  for (const auto& tensor_meta : input_consistent_tensor_metas_) {
    HashCombine(&hash_value, tensor_meta_hash_functor(tensor_meta));
  }
  return hash_value;
}

size_t SrcOpConsistentTensorMetaInferArgs::hash_value() const {
  size_t hash_value = std::hash<AttrMap>()(attrs_);
  hash_value ^= std::hash<Symbol<ParallelDesc>>()(parallel_desc_);
  hash_value ^= std::hash<Symbol<NdSbp>>()(nd_sbp_);
  return hash_value;
}

bool ConsistentTensorMetaInferArgs::operator==(const ConsistentTensorMetaInferArgs& other) const {
  return this->attrs_ == other.attrs_
         && this->input_consistent_tensor_metas_ == other.input_consistent_tensor_metas_;
}

bool SrcOpConsistentTensorMetaInferArgs::operator==(
    const SrcOpConsistentTensorMetaInferArgs& other) const {
  return this->attrs_ == other.attrs_ && this->parallel_desc_ == other.parallel_desc_
         && this->nd_sbp_ == other.nd_sbp_;
}

Maybe<void> ConsistentTensorMetaInferArgs::MakeNdSbpConstraints(
    const UserOpExpr& user_op_expr, NdSbpSignature* nd_sbp_signature) const {
  const auto& input_arg_tuple = *user_op_expr.input_arg_tuple();
  auto* map = nd_sbp_signature->mutable_bn_in_op2nd_sbp();
  for (int i = 0; i < input_arg_tuple.size(); ++i) {
    const auto& constaint = input_consistent_tensor_metas_.at(i).consumer_nd_sbp_constraint();
    if (constaint.has_value()) { (*map)[input_arg_tuple.indexed_bns().at(i)] = *JUST(constaint); }
  }
  return Maybe<void>::Ok();
}

Maybe<void> ConsistentTensorMetaInferArgs::MakeInputBlobDescs(
    const UserOpExpr& user_op_expr, std::vector<BlobDesc>* blob_descs) const {
  CHECK_OR_RETURN(blob_descs->empty());
  const auto& input_arg_tuple = *user_op_expr.input_arg_tuple();
  blob_descs->reserve(input_arg_tuple.size());
  for (int i = 0; i < input_arg_tuple.size(); ++i) {
    const auto& tensor_meta = *input_consistent_tensor_metas_.at(i).tensor_meta();
    const auto& shape = std::const_pointer_cast<Shape>(tensor_meta.shape_ptr());
    blob_descs->emplace_back(shape, tensor_meta.data_type());
  }
  return Maybe<void>::Ok();
}

Maybe<void> ConsistentTensorMetaInferArgs::MakeNdSbpInferHints(
    const UserOpExpr& user_op_expr, const std::vector<BlobDesc>& blob_descs,
    std::vector<NdSbpInferHint>* hints) const {
  CHECK_OR_RETURN(hints->empty());
  const auto& input_arg_tuple = *user_op_expr.input_arg_tuple();
  hints->reserve(input_arg_tuple.size());
  for (int i = 0; i < input_arg_tuple.size(); ++i) {
    const auto& tensor_meta = *input_consistent_tensor_metas_.at(i).tensor_meta();
    const auto* parallel_desc = &*tensor_meta.parallel_desc();
    const auto* blob_desc = &blob_descs.at(i);
    const auto* nd_sbp = &*tensor_meta.nd_sbp();
    hints->emplace_back(parallel_desc, blob_desc, nd_sbp);
  }
  return Maybe<void>::Ok();
}

Maybe<ConsistentTensorMetaInferArgs> ConsistentTensorMetaInferArgs::New(
    const AttrMap& attrs, const TensorTuple& input_tensors) {
  std::shared_ptr<ConsistentTensorMetaInferArgs> infer_args(new ConsistentTensorMetaInferArgs());
  infer_args->attrs_ = attrs;
  infer_args->input_consistent_tensor_metas_.resize(input_tensors.size());
  JUST(infer_args->InitInputConsistentTensorMetas(input_tensors));
  return infer_args;
}

Maybe<SrcOpConsistentTensorMetaInferArgs> SrcOpConsistentTensorMetaInferArgs::New(
    const AttrMap& attrs, Symbol<ParallelDesc> parallel_desc, Symbol<NdSbp> nd_sbp) {
  std::shared_ptr<SrcOpConsistentTensorMetaInferArgs> infer_args(
      new SrcOpConsistentTensorMetaInferArgs());
  infer_args->attrs_ = attrs;
  infer_args->parallel_desc_ = parallel_desc;
  infer_args->nd_sbp_ = nd_sbp;
  return infer_args;
}

Maybe<void> ConsistentTensorMetaInferArgs::InitInputConsistentTensorMetas(
    const TensorTuple& input_tensors) {
  for (int i = 0; i < input_tensors.size(); ++i) {
    const auto& tensor = *input_tensors.at(i);
    const auto& tensor_meta = JUST(tensor.consistent_tensor_meta());
    const auto& constraint = JUST(tensor.consumer_nd_sbp_constraint());
    input_consistent_tensor_metas_.at(i).assign(tensor_meta, constraint);
  }
  return Maybe<void>::Ok();
}

namespace {

Maybe<Operator> MakeOp(const UserOpExpr& user_op_expr, const AttrMap& attrs,
                       const std::string& device_tag) {
  OperatorConf op_conf;
  JUST(user_op_expr.BuildOpConf(&op_conf, attrs));
  DeviceType device_type = JUST(DeviceType4DeviceTag(device_tag));
  return JUST(ConstructOp(op_conf, device_type));
}

Maybe<void> CheckInputParallelDescIdentical(const ConsistentTensorMetaInferArgs& infer_args) {
  if (infer_args.input_consistent_tensor_metas().empty()) { return Maybe<void>::Ok(); }
  const auto& first_parallel_desc =
      infer_args.input_consistent_tensor_metas().begin()->tensor_meta()->parallel_desc();
  for (const auto& input_meta : infer_args.input_consistent_tensor_metas()) {
    CHECK_OR_RETURN(first_parallel_desc == input_meta.tensor_meta()->parallel_desc());
  }
  return Maybe<void>::Ok();
}

Maybe<void> CheckIsDeviceSupportedByOp(const ParallelDesc& parallel_desc,
                                       const std::string& op_type_name) {
  if (IsCpuOnly(op_type_name)) { CHECK_EQ_OR_RETURN(parallel_desc.device_tag(), "cpu"); }
  return Maybe<void>::Ok();
}

class UserOpExprDeviceAndStreamInferContext final : public user_op::DeviceAndStreamInferContext {
 public:
  UserOpExprDeviceAndStreamInferContext(const UserOpExpr* user_op_expr,
                                        const ConsistentTensorMetaInferArgs* infer_args)
      : user_op_expr_(user_op_expr),
        composed_attrs_(infer_args->attrs(), user_op_expr->base_attrs()),
        in_tensor_devices_(user_op_expr_->input_size()),
        out_tensor_devices_(user_op_expr_->output_size()) {
    for (int i = 0; i < user_op_expr_->input_size(); ++i) {
      const auto& parallel_desc =
          infer_args->input_consistent_tensor_metas().at(i).tensor_meta()->parallel_desc();
      in_tensor_devices_.at(i) = CHECK_JUST(GetTensorDevice(parallel_desc));
    }
  }

  const std::vector<std::pair<std::string, int32_t>>& inputs() const override {
    return user_op_expr_->indexed_input_pairs();
  }

  const std::vector<std::pair<std::string, int32_t>>& outputs() const override {
    return user_op_expr_->indexed_output_pairs();
  }

  Symbol<Device>* OutputTensorDevice4ArgNameAndIndex(const std::string& name,
                                                     int64_t index) override {
    const auto& arg_tuple = *user_op_expr_->output_arg_tuple();
    int32_t tuple_index = arg_tuple.TensorTupleIndex4ArgNameAndIndex(name, index);
    CHECK_GE(tuple_index, 0);
    CHECK_LT(tuple_index, user_op_expr_->output_size());
    return &out_tensor_devices_.at(tuple_index);
  }

  Symbol<Device> InputTensorDevice4ArgNameAndIndex(const std::string& name,
                                                   int64_t index) const override {
    const auto& arg_tuple = *user_op_expr_->input_arg_tuple();
    int32_t tuple_index = arg_tuple.TensorTupleIndex4ArgNameAndIndex(name, index);
    CHECK_GE(tuple_index, 0);
    CHECK_LT(tuple_index, user_op_expr_->input_size());
    return in_tensor_devices_.at(tuple_index);
  }

 private:
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return composed_attrs_.Attr4Name(attr_name);
  }
  const UserOpExpr* user_op_expr_;
  const ComposedAttrMap composed_attrs_;
  std::vector<Symbol<Device>> in_tensor_devices_;
  std::vector<Symbol<Device>> out_tensor_devices_;
};

}  // namespace

/* static */ Maybe<Symbol<Stream>> ConsistentTensorInferCache::InferDeviceAndStream(
    const UserOpExpr& user_op_expr, const ConsistentTensorMetaInferArgs& infer_args) {
  if (!user_op_expr.device_and_stream_infer_fn()) {
    Symbol<ParallelDesc> parallel_desc =
        infer_args.input_consistent_tensor_metas().at(0).tensor_meta()->parallel_desc();
    return GetDefaultStreamByPlacement(parallel_desc);
  } else {
    UserOpExprDeviceAndStreamInferContext device_and_stream_ctx(&user_op_expr, &infer_args);
    return TRY(user_op_expr.device_and_stream_infer_fn()(&device_and_stream_ctx));
  }
}

/* static */ Maybe<const ConsistentTensorInferResult> ConsistentTensorInferCache::Infer(
    const UserOpExpr& user_op_expr, const ConsistentTensorMetaInferArgs& infer_args) {
  CHECK_GT_OR_RETURN(infer_args.input_consistent_tensor_metas().size(), 0);
  Symbol<ParallelDesc> parallel_desc =
      infer_args.input_consistent_tensor_metas().at(0).tensor_meta()->parallel_desc();
  JUST(CheckInputParallelDescIdentical(infer_args));
  JUST(CheckIsDeviceSupportedByOp(*parallel_desc, user_op_expr.op_type_name()));
  std::vector<OpArgMutConsistentTensorMeta> output_mut_metas(user_op_expr.output_size());
  {
    // Infer OpArgMutConsistentTensorMeta.
    const auto& input_metas = infer_args.input_consistent_tensor_metas();
    JUST(user_op_expr.InferLogicalShapeAndDType(
        infer_args.attrs(), parallel_desc,
        [&](int32_t i) { return &*input_metas.at(i).tensor_meta(); },
        [&](int32_t i) { return output_mut_metas.at(i).mut_tensor_meta(); }));
  }
  const auto& op = JUST(MakeOp(user_op_expr, infer_args.attrs(), parallel_desc->device_tag()));
  JUST(op->FillOpParallelDesc(parallel_desc.shared_from_symbol()));
  JUST(op->InferParallelSignatureIf());
  {
    // Infer parallel distribution.
    NdSbpSignature nd_sbp_constraints;
    JUST(infer_args.MakeNdSbpConstraints(user_op_expr, &nd_sbp_constraints));
    std::vector<BlobDesc> blob_descs;
    JUST(infer_args.MakeInputBlobDescs(user_op_expr, &blob_descs));
    std::vector<NdSbpInferHint> pd_infer_hints;
    JUST(infer_args.MakeNdSbpInferHints(user_op_expr, blob_descs, &pd_infer_hints));
    const auto& input_arg_tuple = *user_op_expr.input_arg_tuple();
    const auto& NdSbpInferHint4Ibn = [&](const std::string& ibn) -> Maybe<const NdSbpInferHint*> {
      int32_t input_index = input_arg_tuple.bn_in_op2tensor_tuple_index().at(ibn);
      CHECK_GE_OR_RETURN(input_index, 0);
      CHECK_LT_OR_RETURN(input_index, pd_infer_hints.size());
      return &pd_infer_hints.at(input_index);
    };
    // The inferred results can be retrieved by op->NdSbp4BnInOp(obn).
    JUST(op->InferNdSbpSignatureIf(nd_sbp_constraints, *parallel_desc, NdSbpInferHint4Ibn));
  }
  auto result = std::make_unique<ConsistentTensorInferResult>(user_op_expr.input_size(),
                                                              user_op_expr.output_size());
  auto* input_metas = result->mut_input_tensor_metas();
  for (int32_t i = 0; i < user_op_expr.input_size(); ++i) {
    const auto& old_consistent_tensor_meta =
        infer_args.input_consistent_tensor_metas().at(i).tensor_meta();
    const auto& ibn = user_op_expr.input_arg_tuple()->indexed_bns().at(i);
    const auto& nd_sbp = SymbolOf(*JUST(op->NdSbp4BnInOp(ibn)));
    ConsistentTensorMeta consistent_tensor_meta(old_consistent_tensor_meta->shape_ptr(),
                                                old_consistent_tensor_meta->dtype(), nd_sbp,
                                                old_consistent_tensor_meta->parallel_desc());
    input_metas->at(i) = SymbolOf(consistent_tensor_meta);
  }
  auto* output_metas = result->mut_output_tensor_metas();
  for (int32_t i = 0; i < user_op_expr.output_size(); ++i) {
    const auto& output_mut_meta = output_mut_metas.at(i);
    const auto& shape = output_mut_meta.tensor_meta().shape_ptr();
    DataType data_type = output_mut_meta.tensor_meta().data_type();
    const auto& obn = user_op_expr.output_arg_tuple()->indexed_bns().at(i);
    const auto& nd_sbp = SymbolOf(*JUST(op->NdSbp4BnInOp(obn)));
    ConsistentTensorMeta tensor_meta(shape, data_type, nd_sbp, parallel_desc);
    output_metas->at(i) = SymbolOf(tensor_meta);
  }
  result->set_stream(JUST(InferDeviceAndStream(user_op_expr, infer_args)));
  return std::shared_ptr<const ConsistentTensorInferResult>(std::move(result));
}

/* static */ Maybe<const ConsistentTensorInferResult> ConsistentTensorInferCache::Infer(
    const UserOpExpr& user_op_expr, const SrcOpConsistentTensorMetaInferArgs& infer_args) {
  Symbol<ParallelDesc> parallel_desc = infer_args.parallel_desc();
  JUST(CheckIsDeviceSupportedByOp(*parallel_desc, user_op_expr.op_type_name()));
  std::vector<OpArgMutConsistentTensorMeta> output_mut_metas(user_op_expr.output_size());
  {
    // Infer OpArgMutConsistentTensorMeta.
    const auto& GetInputTensorMeta = [](int32_t i) {
      UNIMPLEMENTED();
      return nullptr;
    };
    JUST(user_op_expr.InferLogicalShapeAndDType(
        infer_args.attrs(), parallel_desc, GetInputTensorMeta,
        [&](int32_t i) { return output_mut_metas.at(i).mut_tensor_meta(); }));
  }
  auto result = std::make_unique<ConsistentTensorInferResult>(user_op_expr.input_size(),
                                                              user_op_expr.output_size());
  auto* output_metas = result->mut_output_tensor_metas();
  for (int32_t i = 0; i < user_op_expr.output_size(); ++i) {
    const auto& output_mut_meta = output_mut_metas.at(i);
    const auto& shape = output_mut_meta.tensor_meta().shape_ptr();
    DataType data_type = output_mut_meta.tensor_meta().data_type();
    const auto& nd_sbp = infer_args.nd_sbp();
    ConsistentTensorMeta tensor_meta(shape, data_type, nd_sbp, parallel_desc);
    output_metas->at(i) = SymbolOf(tensor_meta);
  }
  result->set_stream(JUST(GetDefaultStreamByPlacement(parallel_desc)));
  return std::shared_ptr<const ConsistentTensorInferResult>(std::move(result));
}

Maybe<const ConsistentTensorInferResult> ConsistentTensorInferCache::GetOrInfer(
    const ConsistentTensorMetaInferArgs& infer_args) {
  auto iter = cache_.find(infer_args);
  if (iter == cache_.end()) {
    const auto& user_op_expr = user_op_expr_.lock();
    CHECK_OR_RETURN(static_cast<bool>(user_op_expr));
    const auto& output_tensor_metas = JUST(Infer(*user_op_expr, infer_args));
    iter = cache_.emplace(infer_args, output_tensor_metas).first;
  }
  return iter->second;
}

Maybe<const ConsistentTensorInferResult> ConsistentTensorInferCache::GetOrInfer(
    const SrcOpConsistentTensorMetaInferArgs& infer_args) {
  auto iter = src_op_cache_.find(infer_args);
  if (iter == src_op_cache_.end()) {
    const auto& user_op_expr = user_op_expr_.lock();
    CHECK_OR_RETURN(static_cast<bool>(user_op_expr));
    const auto& output_tensor_metas = JUST(Infer(*user_op_expr, infer_args));
    iter = src_op_cache_.emplace(infer_args, output_tensor_metas).first;
  }
  return iter->second;
}

}  // namespace one
}  // namespace oneflow
