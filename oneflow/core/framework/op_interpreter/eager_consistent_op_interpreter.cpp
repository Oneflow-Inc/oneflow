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
  InputConsistentTensorMeta(Symbol<ConsistentTensorMeta> tensor_meta,
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
  ConsistentTensorMetaInferArgs(size_t input_tensor_size, Symbol<PlacementScope> placement_scope,
                                const AttrMap& attrs)
      : input_consistent_tensor_metas_(input_tensor_size),
        placement_scope_(placement_scope),
        attrs_(attrs) {}

  ConsistentTensorMetaInferArgs(const ConsistentTensorMetaInferArgs&) = default;
  ConsistentTensorMetaInferArgs(ConsistentTensorMetaInferArgs&&) = default;
  ~ConsistentTensorMetaInferArgs() = default;

  const std::vector<InputConsistentTensorMeta>& input_consistent_tensor_metas() const {
    return input_consistent_tensor_metas_;
  }
  Symbol<PlacementScope> placement_scope() const { return placement_scope_; }
  const AttrMap& attrs() const { return attrs_; }
  size_t hash_value() const {
    size_t hash_value = std::hash<Symbol<PlacementScope>>()(placement_scope_);
    hash_value ^= std::hash<AttrMap>()(attrs_);
    const auto& tensor_meta_hash_functor = std::hash<InputConsistentTensorMeta>();
    for (const auto& tensor_meta : input_consistent_tensor_metas_) {
      HashCombine(&hash_value, tensor_meta_hash_functor(tensor_meta));
    }
    return hash_value;
  }

  bool operator==(const ConsistentTensorMetaInferArgs& other) const {
    return this->input_consistent_tensor_metas_ == other.input_consistent_tensor_metas_
           && this->placement_scope_ == other.placement_scope_ && this->attrs_ == other.attrs_;
  }

  Maybe<void> InitInputConsistentTensorMetas(const TensorTuple& input_tensors) {
    CHECK_EQ_OR_RETURN(input_consistent_tensor_metas_.size(), input_tensors.size());
    for (int i = 0; i < input_tensors.size(); ++i) {
      const auto& tensor = *input_tensors.at(i);
      const auto& tensor_meta = JUST(tensor.consistent_tensor_meta());
      const auto& constraints = JUST(tensor.consumer_parallel_distribution_constraint());
      input_consistent_tensor_metas_.at(i).assign(tensor_meta, constraints);
    }
    return Maybe<void>::Ok();
  }

 private:
  std::vector<InputConsistentTensorMeta> input_consistent_tensor_metas_;
  Symbol<PlacementScope> placement_scope_;
  AttrMap attrs_;
};

class OpArgConsistentTensorMeta final {
 public:
  OpArgConsistentTensorMeta() : tensor_meta_(), parallel_distribution_() {}
  OpArgConsistentTensorMeta(Symbol<ConsistentTensorMeta> tensor_meta,
                            Symbol<cfg::ParallelDistribution> parallel_distribution)
      : tensor_meta_(tensor_meta),
        parallel_distribution_(parallel_distribution) {}

  OpArgConsistentTensorMeta(const OpArgConsistentTensorMeta&) = default;
  OpArgConsistentTensorMeta(OpArgConsistentTensorMeta&&) = default;
  ~OpArgConsistentTensorMeta() = default;

  size_t hash_value() const {
    return std::hash<Symbol<ConsistentTensorMeta>>()(tensor_meta())
           ^ std::hash<Symbol<cfg::ParallelDistribution>>()(
               parallel_distribution());
  }

  bool operator==(const OpArgConsistentTensorMeta& other) const {
    return this->tensor_meta() == other.tensor_meta()
           && this->parallel_distribution()
                  == other.parallel_distribution();
  }

  Symbol<ConsistentTensorMeta> tensor_meta() const { return tensor_meta_; }
  Symbol<cfg::ParallelDistribution> parallel_distribution() const {
    return parallel_distribution_;
  }
  void assign(Symbol<ConsistentTensorMeta> tensor_meta,
              Symbol<cfg::ParallelDistribution> parallel_distribution) {
    tensor_meta_ = tensor_meta;
    parallel_distribution_ = parallel_distribution;
  }

 private:
  Symbol<ConsistentTensorMeta> tensor_meta_;
  Symbol<cfg::ParallelDistribution> parallel_distribution_;
};

class ConsistentTensorMetaInferResult final {
 public:
  ConsistentTensorMetaInferResult(size_t output_size):
    output_tensors_(std::make_shared<TensorTuple>(output_size)) {}

  const std::shared_ptr<const std::vector<OpArgConsistentTensorMeta>>& output_tensor_meta() const {
    return output_tensor_meta_;
  }
  const std::shared_ptr<TensorTuple> output_tensors() const { return output_tensors(); }

 private:
  std::shared_ptr<const std::vector<OpArgConsistentTensorMeta>> output_tensor_meta_;
  std::shared_ptr<TensorTuple> output_tensors_;
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

  TensorMeta* mut_tensor_meta() { return tensor_meta_; }
  ParallelDistribution* mut_parallel_distribution() { return parallel_distribution_; }

 private:
  TensorMeta tensor_meta_;
  ParallelDistribution parallel_distribution_;
};

std::shared_ptr<const std::vector<OpArgConsistentTensorMeta>> Infer(
    const UserOpExpr& user_op_expr, const ConsistentTensorMetaInferArgs& infer_args) {
  std::vector<OpArgMutConsistentTensorMeta> output_mut_tensor_meta(user_op_expr.output_size());
  TODO();
  return std::shared_ptr<std::vector<OpArgConsistentTensorMeta>>();
}

}

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
