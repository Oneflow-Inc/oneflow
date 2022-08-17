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
#ifndef ONEFLOW_CORE_FRAMEWORK_GLOBAL_TENSOR_INFER_CACHE_H_
#define ONEFLOW_CORE_FRAMEWORK_GLOBAL_TENSOR_INFER_CACHE_H_

#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/stream.h"
#include "oneflow/core/common/tensor_meta.h"
#include "oneflow/core/register/blob_desc.h"
#include "oneflow/core/job/nd_sbp_infer_hint.h"

namespace oneflow {

class NdSbp;

class ParallelDesc;

namespace one {

class GlobalTensorMeta;

class InputGlobalTensorMeta final {
 public:
  InputGlobalTensorMeta() : tensor_meta_(), consumer_nd_sbp_constraint_() {}
  InputGlobalTensorMeta(Symbol<GlobalTensorMeta> tensor_meta,
                        const Optional<Symbol<NdSbp>>& consumer_nd_sbp_constraint)
      : tensor_meta_(tensor_meta), consumer_nd_sbp_constraint_(consumer_nd_sbp_constraint) {}

  InputGlobalTensorMeta(const InputGlobalTensorMeta&) = default;
  InputGlobalTensorMeta(InputGlobalTensorMeta&&) = default;
  ~InputGlobalTensorMeta() = default;

  size_t hash_value() const;
  bool operator==(const InputGlobalTensorMeta& other) const;
  Symbol<GlobalTensorMeta> tensor_meta() const { return tensor_meta_; }
  const Optional<Symbol<NdSbp>>& consumer_nd_sbp_constraint() const {
    return consumer_nd_sbp_constraint_;
  }
  void assign(Symbol<GlobalTensorMeta> tensor_meta,
              const Optional<Symbol<NdSbp>>& consumer_nd_sbp_constraint);

 private:
  Symbol<GlobalTensorMeta> tensor_meta_;
  Optional<Symbol<NdSbp>> consumer_nd_sbp_constraint_;
};

class TensorTuple;
class UserOpExpr;

class GlobalTensorMetaInferArgs final {
 public:
  GlobalTensorMetaInferArgs(const GlobalTensorMetaInferArgs&) = default;
  GlobalTensorMetaInferArgs(GlobalTensorMetaInferArgs&&) = default;
  ~GlobalTensorMetaInferArgs() = default;

  const std::vector<InputGlobalTensorMeta>& input_global_tensor_metas() const {
    return input_global_tensor_metas_;
  }
  const AttrMap& attrs() const { return attrs_; }

  size_t hash_value() const;

  bool operator==(const GlobalTensorMetaInferArgs& other) const;

  Maybe<void> MakeNdSbpConstraints(const UserOpExpr& user_op_expr,
                                   NdSbpSignature* nd_sbp_signature) const;

  Maybe<void> MakeInputBlobDescs(const UserOpExpr& user_op_expr,
                                 std::vector<BlobDesc>* blob_descs) const;

  Maybe<void> MakeNdSbpInferHints(const UserOpExpr& user_op_expr,
                                  const std::vector<BlobDesc>& blob_descs,
                                  std::vector<NdSbpInferHint>* hints) const;

  static Maybe<GlobalTensorMetaInferArgs> New(const AttrMap& attrs,
                                              const TensorTuple& input_tensors);

 private:
  GlobalTensorMetaInferArgs() = default;
  Maybe<void> InitInputGlobalTensorMetas(const TensorTuple& input_tensors);

  AttrMap attrs_;
  std::vector<InputGlobalTensorMeta> input_global_tensor_metas_;
};

class SrcOpGlobalTensorMetaInferArgs final {
 public:
  SrcOpGlobalTensorMetaInferArgs(const SrcOpGlobalTensorMetaInferArgs&) = default;
  SrcOpGlobalTensorMetaInferArgs(SrcOpGlobalTensorMetaInferArgs&&) = default;
  ~SrcOpGlobalTensorMetaInferArgs() = default;

  Symbol<ParallelDesc> parallel_desc() const { return parallel_desc_; }
  Symbol<NdSbp> nd_sbp() const { return nd_sbp_; }
  const AttrMap& attrs() const { return attrs_; }

  size_t hash_value() const;

  bool operator==(const SrcOpGlobalTensorMetaInferArgs& other) const;

  static Maybe<SrcOpGlobalTensorMetaInferArgs> New(const AttrMap& attrs,
                                                   Symbol<ParallelDesc> parallel_desc,
                                                   Symbol<NdSbp> nd_sbp);

 private:
  SrcOpGlobalTensorMetaInferArgs() = default;

  AttrMap attrs_;
  Symbol<ParallelDesc> parallel_desc_;
  Symbol<NdSbp> nd_sbp_;
};

class OpArgMutGlobalTensorMeta final {
 public:
  OpArgMutGlobalTensorMeta()
      : tensor_meta_(std::make_shared<Shape>(), DataType::kInvalidDataType) {}

  OpArgMutGlobalTensorMeta(const OpArgMutGlobalTensorMeta&) = default;
  OpArgMutGlobalTensorMeta(OpArgMutGlobalTensorMeta&&) = default;
  ~OpArgMutGlobalTensorMeta() = default;

  const TensorMeta& tensor_meta() const { return tensor_meta_; }

  TensorMeta* mut_tensor_meta() { return &tensor_meta_; }

 private:
  MutTensorMeta tensor_meta_;
};

}  // namespace one
}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::one::InputGlobalTensorMeta> final {
  size_t operator()(const oneflow::one::InputGlobalTensorMeta& val) const {
    return val.hash_value();
  }
};

template<>
struct hash<oneflow::one::GlobalTensorMetaInferArgs> final {
  size_t operator()(const oneflow::one::GlobalTensorMetaInferArgs& val) const {
    return val.hash_value();
  }
};

template<>
struct hash<oneflow::one::SrcOpGlobalTensorMetaInferArgs> final {
  size_t operator()(const oneflow::one::SrcOpGlobalTensorMetaInferArgs& val) const {
    return val.hash_value();
  }
};

}  // namespace std

namespace oneflow {
namespace one {

class GlobalTensorInferResult final {
 public:
  GlobalTensorInferResult(size_t input_size, size_t output_size)
      : input_tensor_metas_(input_size), output_tensor_metas_(output_size) {}
  GlobalTensorInferResult(const GlobalTensorInferResult&) = delete;
  GlobalTensorInferResult(GlobalTensorInferResult&&) = delete;
  ~GlobalTensorInferResult() = default;

  const std::vector<Symbol<GlobalTensorMeta>>& input_tensor_metas() const {
    return input_tensor_metas_;
  }
  const std::vector<Symbol<GlobalTensorMeta>>& output_tensor_metas() const {
    return output_tensor_metas_;
  }

  std::vector<Symbol<GlobalTensorMeta>>* mut_input_tensor_metas() { return &input_tensor_metas_; }
  std::vector<Symbol<GlobalTensorMeta>>* mut_output_tensor_metas() { return &output_tensor_metas_; }

  const Symbol<Stream>& stream() const { return stream_; }
  void set_stream(const Symbol<Stream>& stream) { stream_ = stream; }

 private:
  std::vector<Symbol<GlobalTensorMeta>> input_tensor_metas_;
  std::vector<Symbol<GlobalTensorMeta>> output_tensor_metas_;
  Symbol<Stream> stream_;
};

class GlobalTensorInferCache final {
 public:
  GlobalTensorInferCache(const std::shared_ptr<const UserOpExpr>& user_op_expr)
      : user_op_expr_(user_op_expr) {}

  Maybe<const GlobalTensorInferResult> GetOrInfer(const GlobalTensorMetaInferArgs& infer_args);

  static Maybe<const GlobalTensorInferResult> Infer(const UserOpExpr& user_op_expr,
                                                    const GlobalTensorMetaInferArgs& infer_args);

  Maybe<const GlobalTensorInferResult> GetOrInfer(const SrcOpGlobalTensorMetaInferArgs& infer_args);

  static Maybe<const GlobalTensorInferResult> Infer(
      const UserOpExpr& user_op_expr, const SrcOpGlobalTensorMetaInferArgs& infer_args);

 private:
  static Maybe<Symbol<Stream>> InferDeviceAndStream(const UserOpExpr& user_op_expr,
                                                    const GlobalTensorMetaInferArgs& infer_args);

  std::weak_ptr<const UserOpExpr> user_op_expr_;
  HashMap<GlobalTensorMetaInferArgs, std::shared_ptr<const GlobalTensorInferResult>> cache_;
  HashMap<SrcOpGlobalTensorMetaInferArgs, std::shared_ptr<const GlobalTensorInferResult>>
      src_op_cache_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_GLOBAL_TENSOR_INFER_CACHE_H_
