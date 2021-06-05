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
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/op_arg_util.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/framework/symbol_storage_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_name_scope.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/job/placement_scope.h"
#include "oneflow/core/eager/foreign_boxing_util.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {
namespace one {

class InputConsistentTensorMeta final {
 public:
  InputConsistentTensorMeta() : tensor_meta_(), consumer_parallel_distribution_constraint_() {}
  InputConsistentTensorMeta(
      Symbol<ConsistentTensorMeta> tensor_meta,
      Symbol<cfg::ParallelDistribution> consumer_parallel_distribution_constraint)
      : tensor_meta_(tensor_meta),
        consumer_parallel_distribution_constraint_(consumer_parallel_distribution_constraint) {}

  InputConsistentTensorMeta(const InputConsistentTensorMeta&) = default;
  InputConsistentTensorMeta(InputConsistentTensorMeta&&) = default;
  ~InputConsistentTensorMeta() = default;

  size_t hash_value() const {
    return std::hash<Symbol<ConsistentTensorMeta>>()(tensor_meta())
           ^ std::hash<Symbol<cfg::ParallelDistribution>>()(
               consumer_parallel_distribution_constraint());
  }

  bool operator==(const InputConsistentTensorMeta& other) const {
    return this->tensor_meta() == other.tensor_meta()
           && this->consumer_parallel_distribution_constraint()
                  == other.consumer_parallel_distribution_constraint();
  }

  Symbol<ConsistentTensorMeta> tensor_meta() const { return tensor_meta_; }
  Symbol<cfg::ParallelDistribution> consumer_parallel_distribution_constraint() const {
    return consumer_parallel_distribution_constraint_;
  }
  void assign(Symbol<ConsistentTensorMeta> tensor_meta,
              Symbol<cfg::ParallelDistribution> consumer_parallel_distribution_constraint) {
    tensor_meta_ = tensor_meta;
    consumer_parallel_distribution_constraint_ = consumer_parallel_distribution_constraint;
  }

 private:
  Symbol<ConsistentTensorMeta> tensor_meta_;
  Symbol<cfg::ParallelDistribution> consumer_parallel_distribution_constraint_;
};

}  // namespace one
}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::one::InputConsistentTensorMeta> final {
  size_t operator()(const oneflow::one::InputConsistentTensorMeta& val) const {
    return val.hash_value();
  }
};

}  // namespace std

namespace oneflow {
namespace one {

class ConsistentTensorMetaInferArgs final {
 public:
  ConsistentTensorMetaInferArgs() = default;
  ConsistentTensorMetaInferArgs(const ConsistentTensorMetaInferArgs&) = default;
  ConsistentTensorMetaInferArgs(ConsistentTensorMetaInferArgs&&) = default;
  ~ConsistentTensorMetaInferArgs() = default;

  Maybe<void> Init(const TensorTuple& input_tensors, Symbol<PlacementScope> placement_scope,
                   const AttrMap& attrs, const std::string& current_scope_device_tag) {
    input_consistent_tensor_metas_.resize(input_tensors.size());
    placement_scope_ = placement_scope;
    attrs_ = attrs;
    current_scope_device_tag_ = current_scope_device_tag;
    JUST(InitInputConsistentTensorMetas(input_tensors));
    return Maybe<void>::Ok();
  }

  const std::vector<InputConsistentTensorMeta>& input_consistent_tensor_metas() const {
    return input_consistent_tensor_metas_;
  }
  Symbol<PlacementScope> placement_scope() const { return placement_scope_; }
  const AttrMap& attrs() const { return attrs_; }
  const std::string& current_scope_device_tag() const { return current_scope_device_tag_; }
  size_t hash_value() const {
    size_t hash_value = std::hash<Symbol<PlacementScope>>()(placement_scope_);
    hash_value ^= std::hash<AttrMap>()(attrs_);
    const auto& tensor_meta_hash_functor = std::hash<InputConsistentTensorMeta>();
    for (const auto& tensor_meta : input_consistent_tensor_metas_) {
      HashCombine(&hash_value, tensor_meta_hash_functor(tensor_meta));
    }
    hash_value ^= std::hash<std::string>()(current_scope_device_tag_);
    return hash_value;
  }

  bool operator==(const ConsistentTensorMetaInferArgs& other) const {
    return this->input_consistent_tensor_metas_ == other.input_consistent_tensor_metas_
           && this->placement_scope_ == other.placement_scope_ && this->attrs_ == other.attrs_
           && this->current_scope_device_tag_ == other.current_scope_device_tag_;
  }

  Maybe<void> MakeParallelDistributionConstraints(
      const UserOpExpr& user_op_expr,
      cfg::ParallelDistributionSignature* parallel_distribution_signature) const {
    const auto& input_arg_tuple = *user_op_expr.input_arg_tuple();
    auto* map = parallel_distribution_signature->mutable_bn_in_op2parallel_distribution();
    for (int i = 0; i < input_arg_tuple.size(); ++i) {
      (*map)[input_arg_tuple.indexed_bns().at(i)] =
          *input_consistent_tensor_metas_.at(i).consumer_parallel_distribution_constraint();
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> MakeInputBlobDescs(const UserOpExpr& user_op_expr,
                                 std::vector<BlobDesc>* blob_descs) const {
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

  Maybe<void> MakeParallelDistributionInferHints(
      const UserOpExpr& user_op_expr, const std::vector<BlobDesc>& blob_descs,
      std::vector<ParallelDistributionInferHint>* hints) const {
    CHECK_OR_RETURN(hints->empty());
    const auto& input_arg_tuple = *user_op_expr.input_arg_tuple();
    hints->reserve(input_arg_tuple.size());
    for (int i = 0; i < input_arg_tuple.size(); ++i) {
      const auto& tensor_meta = *input_consistent_tensor_metas_.at(i).tensor_meta();
      const auto* parallel_desc = &*tensor_meta.parallel_desc();
      const auto* blob_desc = &blob_descs.at(i);
      const auto* parallel_distribution = &*tensor_meta.parallel_distribution();
      hints->emplace_back(parallel_desc, blob_desc, parallel_distribution);
    }
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InitInputConsistentTensorMetas(const TensorTuple& input_tensors) {
    for (int i = 0; i < input_tensors.size(); ++i) {
      const auto& tensor = *input_tensors.at(i);
      const auto& tensor_meta = JUST(tensor.consistent_tensor_meta());
      const auto& constraints = JUST(tensor.consumer_parallel_distribution_constraint());
      input_consistent_tensor_metas_.at(i).assign(tensor_meta, constraints);
      return Maybe<void>::Ok();
    }
  }

  std::vector<InputConsistentTensorMeta> input_consistent_tensor_metas_;
  Symbol<PlacementScope> placement_scope_;
  AttrMap attrs_;
  std::string current_scope_device_tag_;
};

}  // namespace one
}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::one::ConsistentTensorMetaInferArgs> final {
  size_t operator()(const oneflow::one::ConsistentTensorMetaInferArgs& val) const {
    return val.hash_value();
  }
};

}  // namespace std

namespace oneflow {
namespace one {

namespace {

class OpArgMutConsistentTensorMeta final {
 public:
  OpArgMutConsistentTensorMeta()
      : tensor_meta_(std::make_shared<Shape>(), DataType::kInvalidDataType),
        parallel_distribution_() {}

  OpArgMutConsistentTensorMeta(const OpArgMutConsistentTensorMeta&) = default;
  OpArgMutConsistentTensorMeta(OpArgMutConsistentTensorMeta&&) = default;
  ~OpArgMutConsistentTensorMeta() = default;

  const TensorMeta& tensor_meta() const { return tensor_meta_; }
  const cfg::ParallelDistribution& parallel_distribution() const { return parallel_distribution_; }

  TensorMeta* mut_tensor_meta() { return &tensor_meta_; }
  cfg::ParallelDistribution* mut_parallel_distribution() { return &parallel_distribution_; }

 private:
  TensorMeta tensor_meta_;
  cfg::ParallelDistribution parallel_distribution_;
};

Maybe<Operator> MakeOp(const UserOpExpr& user_op_expr, const AttrMap& attrs,
                       const std::string& device_tag) {
  OperatorConf op_conf;
  JUST(user_op_expr.BuildOpConf(&op_conf, attrs));
  DeviceType device_type = JUST(DeviceType4DeviceTag(device_tag));
  return ConstructOp(op_conf, device_type);
}

Maybe<const std::vector<Symbol<ConsistentTensorMeta>>> Infer(
    const UserOpExpr& user_op_expr, const ConsistentTensorMetaInferArgs& infer_args) {
  Symbol<ParallelDesc> parallel_desc;
  {
    // Get parallel description.
    const auto& placement_scope = infer_args.placement_scope();
    const auto& dev_tag = infer_args.current_scope_device_tag();
    const auto& op_type_name = user_op_expr.op_type_name();
    parallel_desc = JUST(placement_scope->GetParallelDesc(dev_tag, op_type_name));
  }
  std::vector<OpArgMutConsistentTensorMeta> output_mut_metas(user_op_expr.output_size());
  {
    // Infer OpArgMutConsistentTensorMeta.
    const auto& input_metas = infer_args.input_consistent_tensor_metas();
    JUST(user_op_expr.InferLogicalShapeAndDType(
        infer_args.attrs(), parallel_desc->device_tag(),
        [&](int32_t i) { return &*input_metas.at(i).tensor_meta(); },
        [&](int32_t i) { return output_mut_metas.at(i).mut_tensor_meta(); }));
  }
  const auto& op = JUST(MakeOp(user_op_expr, infer_args.attrs(), parallel_desc->device_tag()));
  {
    // Infer parallel distribution.
    cfg::ParallelDistributionSignature parallel_distribution_constraints;
    JUST(infer_args.MakeParallelDistributionConstraints(user_op_expr,
                                                        &parallel_distribution_constraints));
    std::vector<BlobDesc> blob_descs;
    JUST(infer_args.MakeInputBlobDescs(user_op_expr, &blob_descs));
    std::vector<ParallelDistributionInferHint> pd_infer_hints;
    JUST(infer_args.MakeParallelDistributionInferHints(user_op_expr, blob_descs, &pd_infer_hints));
    const auto& input_arg_tuple = *user_op_expr.input_arg_tuple();
    const auto& ParallelDistributionInferHint4Ibn =
        [&](const std::string& ibn) -> Maybe<const ParallelDistributionInferHint*> {
      int32_t input_index = input_arg_tuple.bn_in_op2tensor_tuple_index().at(ibn);
      CHECK_GE_OR_RETURN(input_index, 0);
      CHECK_LT_OR_RETURN(input_index, pd_infer_hints.size());
      return &pd_infer_hints.at(input_index);
    };
    // The inferred results can be retrieved by op->ParallelDistribution4BnInOp(obn).
    JUST(op->InferParallelDistributionSignatureIf(parallel_distribution_constraints, *parallel_desc,
                                                  ParallelDistributionInferHint4Ibn));
  }
  auto* output_metas = new std::vector<Symbol<ConsistentTensorMeta>>(user_op_expr.output_size());
  for (int32_t i = 0; i < user_op_expr.output_size(); ++i) {
    const auto& output_mut_meta = output_mut_metas.at(i);
    const auto& shape = output_mut_meta.tensor_meta().shape_ptr();
    DataType data_type = output_mut_meta.tensor_meta().data_type();
    const auto& obn = user_op_expr.output_arg_tuple()->indexed_bns().at(i);
    const auto& parallel_distribution = SymbolOf(*JUST(op->ParallelDistribution4BnInOp(obn)));
    ConsistentTensorMeta tensor_meta(shape, data_type, parallel_distribution, parallel_desc);
    output_metas->at(i) = SymbolOf(tensor_meta);
  }
  return std::shared_ptr<const std::vector<Symbol<ConsistentTensorMeta>>>(output_metas);
}

}  // namespace

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const UserOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const VariableOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const CastToMirroredOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const CastFromMirroredOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const DistributeSplitOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const DistributeCloneOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const DistributeConcatOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerConsistentInterpreter::ApplyImpl(const DistributeAddOpExpr& op_expr,
                                                  const TensorTuple& inputs, TensorTuple* outputs,
                                                  const AttrMap& attrs) const {
  OF_UNIMPLEMENTED();
}

}  // namespace one
}  // namespace oneflow
