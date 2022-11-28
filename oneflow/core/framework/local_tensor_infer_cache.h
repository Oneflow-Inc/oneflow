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
#ifndef ONEFLOW_CORE_FRAMEWORK_LOCAL_TENSOR_INFER_CACHE_H_
#define ONEFLOW_CORE_FRAMEWORK_LOCAL_TENSOR_INFER_CACHE_H_

#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/op_args_vector.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/stream.h"
#include "oneflow/core/common/tensor_meta.h"

namespace oneflow {

class Device;

namespace one {

class TensorTuple;
class UserOpExpr;

class LocalTensorMetaInferArgs final {
 public:
  LocalTensorMetaInferArgs() = default;
  LocalTensorMetaInferArgs(const LocalTensorMetaInferArgs&) = default;
  LocalTensorMetaInferArgs(LocalTensorMetaInferArgs&&) = default;
  ~LocalTensorMetaInferArgs() = default;

  const OpArgsVector<Symbol<LocalTensorMeta>>& input_local_tensor_metas() const {
    return input_local_tensor_metas_;
  }
  const AttrMap& attrs() const { return attrs_; }

  const Symbol<Device>& default_device() const { return default_device_; }

  size_t hash_value() const;

  bool operator==(const LocalTensorMetaInferArgs& other) const;

  Maybe<void> Init(const AttrMap& attrs, Symbol<Device> default_device,
                   const TensorTuple& input_tensors);

 private:
  Maybe<void> InitInputLocalTensorMetas(const TensorTuple& input_tensors);

  AttrMap attrs_;
  Symbol<Device> default_device_;
  OpArgsVector<Symbol<LocalTensorMeta>> input_local_tensor_metas_;
};

}  // namespace one
}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::one::LocalTensorMetaInferArgs> final {
  size_t operator()(const oneflow::one::LocalTensorMetaInferArgs& val) const {
    return val.hash_value();
  }
};

}  // namespace std

namespace oneflow {
namespace one {

class LocalTensorInferResult final {
 public:
  LocalTensorInferResult(size_t output_size) : output_tensor_metas_(output_size) {}
  LocalTensorInferResult(const LocalTensorInferResult&) = delete;
  LocalTensorInferResult(LocalTensorInferResult&&) = delete;
  ~LocalTensorInferResult() = default;

  const OpArgsVector<Symbol<LocalTensorMeta>>& output_tensor_metas() const {
    return output_tensor_metas_;
  }
  OpArgsVector<Symbol<LocalTensorMeta>>* mut_output_tensor_metas() { return &output_tensor_metas_; }

  const Symbol<Stream>& stream() const { return stream_; }
  void set_stream(const Symbol<Stream>& stream) { stream_ = stream; }

 private:
  OpArgsVector<Symbol<LocalTensorMeta>> output_tensor_metas_;
  Symbol<Stream> stream_;
};

class LocalTensorInferCache final {
 public:
  LocalTensorInferCache(const std::shared_ptr<const UserOpExpr>& user_op_expr)
      : user_op_expr_(user_op_expr) {}

  Maybe<const LocalTensorInferResult> GetOrInfer(const LocalTensorMetaInferArgs& infer_args);

 private:
  static Maybe<const LocalTensorInferResult> Infer(const UserOpExpr& user_op_expr,
                                                   const LocalTensorMetaInferArgs& infer_args);

  std::weak_ptr<const UserOpExpr> user_op_expr_;
  HashMap<LocalTensorMetaInferArgs, std::shared_ptr<const LocalTensorInferResult>> cache_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_LOCAL_TENSOR_INFER_CACHE_H_
