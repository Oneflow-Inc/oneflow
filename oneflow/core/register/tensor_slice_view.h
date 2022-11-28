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
#ifndef ONEFLOW_CORE_REGISTER_TENSOR_SLICE_VIEW_H_
#define ONEFLOW_CORE_REGISTER_TENSOR_SLICE_VIEW_H_

#include "oneflow/core/common/range.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/nd_index.h"
#include "oneflow/core/register/tensor_slice_view.pb.h"

namespace oneflow {

class TensorSliceView final {
 public:
  TensorSliceView() = default;
  TensorSliceView(const std::initializer_list<Range>& ranges);
  explicit TensorSliceView(const std::vector<Range>& ranges);
  explicit TensorSliceView(const TensorSliceViewProto& proto);
  explicit TensorSliceView(const Shape& shape);

  TensorSliceView& operator=(const TensorSliceView& other);
  bool operator==(const TensorSliceView& rhs) const;
  bool operator!=(const TensorSliceView& rhs) const;

  bool IsEmpty() const;
  TensorSliceView Intersect(const TensorSliceView& other) const;
  bool Contains(const TensorSliceView& other) const;
  const Range& At(int64_t index) const;
  const Shape& shape() const;
  const std::vector<Range>& range_vec() const;
  size_t NumAxes() const;
  NdIndex OffsetTo(const TensorSliceView& other) const;
  void ToProto(TensorSliceViewProto* proto) const;

  static TensorSliceView Concatenate(std::vector<TensorSliceView>& slices, int64_t axis);

 private:
  std::vector<Range> range_vec_;
  Shape shape_;

  void UpdateShape();
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_TENSOR_SLICE_VIEW_H_
