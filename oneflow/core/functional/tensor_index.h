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

#ifndef ONEFLOW_CORE_FUNCTIONAL_TENSOR_INDEX_H_
#define ONEFLOW_CORE_FUNCTIONAL_TENSOR_INDEX_H_

#include <cstdint>
#include <limits>
#include <vector>

#include "oneflow/core/common/shape.h"

namespace oneflow {
namespace one {

class Tensor;
class TensorTuple;

namespace functional {

namespace detail {

struct NoneIndex {};
struct EllipsisIndex {};

class Slice {
 public:
  Slice() : Slice(0, std::numeric_limits<int64_t>::max(), 1) {}
  explicit Slice(int64_t start) : Slice(start, std::numeric_limits<int64_t>::max(), 1) {}
  explicit Slice(int64_t start, int64_t end) : Slice(start, end, 1) {}
  explicit Slice(int64_t start, int64_t end, int64_t step)
      : start_(start), end_(end), step_(step) {}

  int64_t start() const { return start_; }
  int64_t end() const { return end_; }
  int64_t step() const { return step_; }
  std::string ToString() const {
    std::stringstream ss;
    ss << "[" << start_ << ":" << end_ << ":" << step_ << "]\n";
    return ss.str();
  }

 private:
  int64_t start_;
  int64_t end_;
  int64_t step_;
};

class IndexItem {
 public:
  IndexItem() : IndexItem(NoneIndex()) {}
  explicit IndexItem(NoneIndex none) : item_{.dummy = 0}, tag_(HAS_NONE) {}

  explicit IndexItem(int64_t start, int64_t end, int64_t step)
      : item_{.slice = Slice{start, end, step}}, tag_(HAS_SLICE) {}
  explicit IndexItem(const Slice& slice) : item_{.slice = slice}, tag_(HAS_SLICE) {}

  explicit IndexItem(int64_t index) : item_{.i = index}, tag_(HAS_INT) {}
  explicit IndexItem(bool boolean) : item_{.b = boolean}, tag_(HAS_BOOLEAN) {}
  explicit IndexItem(EllipsisIndex ellipsis) : item_{.dummy = 0}, tag_(HAS_ELLIPSIS) {}

  explicit IndexItem(const std::shared_ptr<Tensor>& tensor)
      : item_{.dummy = 0}, tensor_(tensor), tag_(HAS_TENSOR) {}

  bool IsSlice() const { return tag_ == HAS_SLICE; }
  const Slice& slice() const { return item_.slice; }

  bool IsInteger() const { return tag_ == HAS_INT; }
  int64_t integer() const { return item_.i; }

  bool IsBoolean() const { return tag_ == HAS_BOOLEAN; }
  bool boolean() const { return item_.b; }

  bool IsEllipsis() const { return tag_ == HAS_ELLIPSIS; }

  bool IsNone() const { return tag_ == HAS_NONE; }

  bool IsTensor() const { return tag_ == HAS_TENSOR; }
  const std::shared_ptr<Tensor>& tensor() const { return tensor_; }

 private:
  union {
    Slice slice;
    bool b;
    int64_t i;
    char dummy;
  } item_;
  std::shared_ptr<Tensor> tensor_;
  enum { HAS_SLICE, HAS_BOOLEAN, HAS_INT, HAS_ELLIPSIS, HAS_NONE, HAS_TENSOR } tag_;
};

}  // namespace detail

class TensorIndex : public std::vector<detail::IndexItem> {
 public:
  using std::vector<detail::IndexItem>::vector;
};

bool IsMaskTensor(const std::shared_ptr<Tensor>& tensor);

Maybe<void> PrepareSliceIndices(const TensorIndex& index, const Shape& shape,
                                std::vector<detail::Slice>* slice_indices,
                                TensorTuple* tensor_indices, std::vector<int64_t>* expand_dims,
                                std::vector<int64_t>* target_dims);

Maybe<std::vector<detail::Slice>> RemoveExpandDimSlice(
    const std::vector<detail::Slice>& expand_slices, const std::vector<int64_t>& expand_dims);

Maybe<Tensor> ApplyAdvancedIndexing(const std::shared_ptr<Tensor>& input,
                                    const TensorTuple& indices);

Maybe<Tensor> ApplySelectIndexing(const std::shared_ptr<one::Tensor>& input,
                                  const TensorIndex& index);

Maybe<void> UnifyInputAndIndicesOnDevice(const std::shared_ptr<Tensor>& x,
                                         TensorTuple& tensor_indices);

Maybe<Tensor> ApplyAdvancedIndexingUpdate(const std::shared_ptr<Tensor>& input,
                                          const TensorTuple& indices,
                                          const std::shared_ptr<Tensor>& value);

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FUNCTIONAL_TENSOR_INDEX_H_
