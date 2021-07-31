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
#ifndef ONEFLOW_CORE_FRAMEWORK_CONSISTENT_TENSOR_INFER_CACHE_H_
#define ONEFLOW_CORE_FRAMEWORK_CONSISTENT_TENSOR_INFER_CACHE_H_

#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/tensor_meta.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"
#include "oneflow/core/job/parallel_distribution_infer_hint.h"

namespace oneflow {

namespace cfg {
class ParallelDistribution;
}

class ParallelDesc;

namespace one {

class ConsistentTensorMeta;

class InputConsistentTensorMeta final {
 public:
  InputConsistentTensorMeta() : tensor_meta_(), consumer_parallel_distribution_constraint_() {}
  InputConsistentTensorMeta(
      Symbol<ConsistentTensorMeta> tensor_meta,
      const Optional<Symbol<cfg::ParallelDistribution>>& consumer_parallel_distribution_constraint)
      : tensor_meta_(tensor_meta),
        consumer_parallel_distribution_constraint_(consumer_parallel_distribution_constraint) {}

  InputConsistentTensorMeta(const InputConsistentTensorMeta&) = default;
  InputConsistentTensorMeta(InputConsistentTensorMeta&&) = default;
  ~InputConsistentTensorMeta() = default;

  size_t hash_value() const;
  bool operator==(const InputConsistentTensorMeta& other) const;
  Symbol<ConsistentTensorMeta> tensor_meta() const { return tensor_meta_; }
  const Optional<Symbol<cfg::ParallelDistribution>>& consumer_parallel_distribution_constraint()
      const {
    return consumer_parallel_distribution_constraint_;
  }
  void assign(
      Symbol<ConsistentTensorMeta> tensor_meta,
      const Optional<Symbol<cfg::ParallelDistribution>>& consumer_parallel_distribution_constraint);

 private:
  Symbol<ConsistentTensorMeta> tensor_meta_;
  Optional<Symbol<cfg::ParallelDistribution>> consumer_parallel_distribution_constraint_;
};

class TensorTuple;
class UserOpExpr;

class ConsistentTensorMetaInferArgs final {
 public:
  ConsistentTensorMetaInferArgs(const ConsistentTensorMetaInferArgs&) = default;
  ConsistentTensorMetaInferArgs(ConsistentTensorMetaInferArgs&&) = default;
  ~ConsistentTensorMetaInferArgs() = default;

  const std::vector<InputConsistentTensorMeta>& input_consistent_tensor_metas() const {
    return input_consistent_tensor_metas_;
  }
  const AttrMap& attrs() const { return attrs_; }

  size_t hash_value() const;

  bool operator==(const ConsistentTensorMetaInferArgs& other) const;

  Maybe<void> MakeParallelDistributionConstraints(
      const UserOpExpr& user_op_expr,
      cfg::ParallelDistributionSignature* parallel_distribution_signature) const;

  Maybe<void> MakeInputBlobDescs(const UserOpExpr& user_op_expr,
                                 std::vector<BlobDesc>* blob_descs) const;

  Maybe<void> MakeParallelDistributionInferHints(
      const UserOpExpr& user_op_expr, const std::vector<BlobDesc>& blob_descs,
      std::vector<ParallelDistributionInferHint>* hints) const;

  static Maybe<ConsistentTensorMetaInferArgs> New(const AttrMap& attrs,
                                                  const TensorTuple& input_tensors);

 private:
  ConsistentTensorMetaInferArgs() = default;
  Maybe<void> InitInputConsistentTensorMetas(const TensorTuple& input_tensors);

  AttrMap attrs_;
  std::vector<InputConsistentTensorMeta> input_consistent_tensor_metas_;
};

class SrcOpConsistentTensorMetaInferArgs final {
 public:
  SrcOpConsistentTensorMetaInferArgs(const SrcOpConsistentTensorMetaInferArgs&) = default;
  SrcOpConsistentTensorMetaInferArgs(SrcOpConsistentTensorMetaInferArgs&&) = default;
  ~SrcOpConsistentTensorMetaInferArgs() = default;

  Symbol<ParallelDesc> parallel_desc() const { return parallel_desc_; }
  Symbol<cfg::ParallelDistribution> parallel_distribution() const { return parallel_distribution_; }
  const AttrMap& attrs() const { return attrs_; }

  size_t hash_value() const;

  bool operator==(const SrcOpConsistentTensorMetaInferArgs& other) const;

  static Maybe<SrcOpConsistentTensorMetaInferArgs> New(
      const AttrMap& attrs, Symbol<ParallelDesc> parallel_desc,
      Symbol<cfg::ParallelDistribution> parallel_distribution);

 private:
  SrcOpConsistentTensorMetaInferArgs() = default;

  AttrMap attrs_;
  Symbol<ParallelDesc> parallel_desc_;
  Symbol<cfg::ParallelDistribution> parallel_distribution_;
};

class OpArgMutConsistentTensorMeta final {
 public:
  OpArgMutConsistentTensorMeta()
      : tensor_meta_(std::make_shared<Shape>(), DataType::kInvalidDataType) {}

  OpArgMutConsistentTensorMeta(const OpArgMutConsistentTensorMeta&) = default;
  OpArgMutConsistentTensorMeta(OpArgMutConsistentTensorMeta&&) = default;
  ~OpArgMutConsistentTensorMeta() = default;

  const TensorMeta& tensor_meta() const { return tensor_meta_; }

  TensorMeta* mut_tensor_meta() { return &tensor_meta_; }

 private:
  TensorMeta tensor_meta_;
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

template<>
struct hash<oneflow::one::ConsistentTensorMetaInferArgs> final {
  size_t operator()(const oneflow::one::ConsistentTensorMetaInferArgs& val) const {
    return val.hash_value();
  }
};

template<>
struct hash<oneflow::one::SrcOpConsistentTensorMetaInferArgs> final {
  size_t operator()(const oneflow::one::SrcOpConsistentTensorMetaInferArgs& val) const {
    return val.hash_value();
  }
};

}  // namespace std

namespace oneflow {
namespace one {

class ConsistentTensorInferResult final {
 public:
  ConsistentTensorInferResult(size_t input_size, size_t output_size)
      : input_parallel_distributions_(input_size), output_tensor_metas_(output_size) {}
  ConsistentTensorInferResult(const ConsistentTensorInferResult&) = delete;
  ConsistentTensorInferResult(ConsistentTensorInferResult&&) = delete;
  ~ConsistentTensorInferResult() = default;

  const std::vector<Symbol<cfg::ParallelDistribution>>& input_parallel_distributions() const {
    return input_parallel_distributions_;
  }
  const std::vector<Symbol<ConsistentTensorMeta>>& output_tensor_metas() const {
    return output_tensor_metas_;
  }

  std::vector<Symbol<cfg::ParallelDistribution>>* mut_input_parallel_distributions() {
    return &input_parallel_distributions_;
  }
  std::vector<Symbol<ConsistentTensorMeta>>* mut_output_tensor_metas() {
    return &output_tensor_metas_;
  }

 private:
  std::vector<Symbol<cfg::ParallelDistribution>> input_parallel_distributions_;
  std::vector<Symbol<ConsistentTensorMeta>> output_tensor_metas_;
};

class ConsistentTensorInferCache final {
 public:
  ConsistentTensorInferCache(const std::shared_ptr<const UserOpExpr>& user_op_expr)
      : user_op_expr_(user_op_expr) {}

  Maybe<const ConsistentTensorInferResult> GetOrInfer(
      const ConsistentTensorMetaInferArgs& infer_args);

  static Maybe<const ConsistentTensorInferResult> Infer(
      const UserOpExpr& user_op_expr, const ConsistentTensorMetaInferArgs& infer_args);

  Maybe<const ConsistentTensorInferResult> GetOrInfer(
      const SrcOpConsistentTensorMetaInferArgs& infer_args);

  static Maybe<const ConsistentTensorInferResult> Infer(
      const UserOpExpr& user_op_expr, const SrcOpConsistentTensorMetaInferArgs& infer_args);

 private:
  std::weak_ptr<const UserOpExpr> user_op_expr_;
  HashMap<ConsistentTensorMetaInferArgs, std::shared_ptr<const ConsistentTensorInferResult>> cache_;
  HashMap<SrcOpConsistentTensorMetaInferArgs, std::shared_ptr<const ConsistentTensorInferResult>>
      src_op_cache_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_CONSISTENT_TENSOR_INFER_CACHE_H_
