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
#include "oneflow/core/register/tensor_slice_view.h"

namespace oneflow {

TensorSliceView::TensorSliceView(const std::initializer_list<Range>& ranges) : range_vec_(ranges) {
  UpdateShape();
}

TensorSliceView::TensorSliceView(const std::vector<Range>& ranges) : range_vec_(ranges) {
  UpdateShape();
}

TensorSliceView::TensorSliceView(const TensorSliceViewProto& proto) {
  range_vec_.resize(proto.dim_size());
  std::transform(proto.dim().cbegin(), proto.dim().cend(), range_vec_.begin(),
                 [](const RangeProto& rp) { return Range(rp); });
  UpdateShape();
}

TensorSliceView::TensorSliceView(const Shape& shape) {
  range_vec_.resize(shape.dim_vec().size());
  std::transform(shape.dim_vec().cbegin(), shape.dim_vec().cend(), range_vec_.begin(),
                 [](const int64_t dim_size) { return Range(0, dim_size); });
  UpdateShape();
}

TensorSliceView& TensorSliceView::operator=(const TensorSliceView& other) {
  range_vec_ = other.range_vec_;
  UpdateShape();
  return *this;
}

bool TensorSliceView::operator==(const TensorSliceView& rhs) const {
  return range_vec_ == rhs.range_vec_;
}

bool TensorSliceView::operator!=(const TensorSliceView& rhs) const { return !(*this == rhs); }

void TensorSliceView::UpdateShape() {
  DimVector dim_vec(range_vec_.size());
  std::transform(range_vec_.cbegin(), range_vec_.cend(), dim_vec.begin(),
                 [](const Range& range) { return range.size(); });
  shape_ = Shape(dim_vec);
}

bool TensorSliceView::IsEmpty() const { return range_vec_.empty(); }

bool TensorSliceView::Contains(const TensorSliceView& other) const {
  if (other.IsEmpty()) { return true; }
  CHECK_EQ(NumAxes(), other.NumAxes());
  FOR_RANGE(int64_t, i, 0, NumAxes()) {
    if (range_vec_.at(i).begin() > other.range_vec_.at(i).begin()
        || range_vec_.at(i).end() < other.range_vec_.at(i).end()) {
      return false;
    }
  }
  return true;
}

TensorSliceView TensorSliceView::Intersect(const TensorSliceView& other) const {
  if (IsEmpty() || other.IsEmpty()) { return TensorSliceView(); }
  CHECK_EQ(other.range_vec_.size(), range_vec_.size());
  std::vector<Range> intersection_vec;
  intersection_vec.reserve(range_vec_.size());
  const Range zero(0, 0);
  FOR_RANGE(int64_t, i, 0, range_vec_.size()) {
    const Range intersection = FindIntersectant(range_vec_.at(i), other.range_vec_.at(i));
    if (intersection == zero) {
      return TensorSliceView();
    } else {
      intersection_vec.emplace_back(intersection);
    }
  }
  return TensorSliceView(intersection_vec);
}

const Range& TensorSliceView::At(int64_t index) const { return range_vec_.at(index); }

const Shape& TensorSliceView::shape() const { return shape_; }

const std::vector<Range>& TensorSliceView::range_vec() const { return range_vec_; }

size_t TensorSliceView::NumAxes() const { return range_vec_.size(); }

NdIndex TensorSliceView::OffsetTo(const TensorSliceView& other) const {
  CHECK_EQ(other.NumAxes(), NumAxes());
  DimVector indices_vec(range_vec_.size());
  std::transform(range_vec_.cbegin(), range_vec_.cend(), other.range_vec_.cbegin(),
                 indices_vec.begin(),
                 [](const Range& lhs, const Range& rhs) { return lhs.begin() - rhs.begin(); });
  return NdIndex(indices_vec);
}

void TensorSliceView::ToProto(TensorSliceViewProto* proto) const {
  for (const Range& range : range_vec_) { range.ToProto(proto->mutable_dim()->Add()); }
}

TensorSliceView TensorSliceView::Concatenate(std::vector<TensorSliceView>& slices, int64_t axis) {
  CHECK_GT(slices.size(), 0);
  const int64_t num_axes = slices.front().shape().NumAxes();
  FOR_RANGE(int64_t, i, 1, slices.size()) { CHECK_EQ(slices.at(i).NumAxes(), num_axes); }
  CHECK_GE(axis, 0);
  CHECK_LT(axis, num_axes);
  FOR_RANGE(int64_t, i, 0, num_axes) {
    if (i == axis) {
      CHECK(std::adjacent_find(slices.cbegin(), slices.cend() - 1,
                               [&](const TensorSliceView& lhs, const TensorSliceView& rhs) {
                                 return lhs.At(i).end() != rhs.At(i).begin();
                               })
            == slices.cend() - 1);
    } else {
      const Range& dim_range = slices.front().At(i);
      CHECK(std::all_of(slices.cbegin() + 1, slices.cbegin(),
                        [&](const TensorSliceView& view) { return view.At(i) == dim_range; }));
    }
  }
  std::vector<Range> range_vec = slices.front().range_vec();
  range_vec.at(axis).mut_end() = slices.back().At(axis).end();
  return TensorSliceView(range_vec);
}

}  // namespace oneflow
