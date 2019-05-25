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
  std::vector<int64_t> dim_vec(range_vec_.size());
  std::transform(range_vec_.cbegin(), range_vec_.cend(), dim_vec.begin(),
                 [](const Range& range) { return range.size(); });
  shape_ = Shape(dim_vec);
}

bool TensorSliceView::IsEmpty() const { return range_vec_.empty(); }

TensorSliceView TensorSliceView::Intersect(const TensorSliceView& other) const {
  CHECK_EQ(other.range_vec_.size(), range_vec_.size());
  std::vector<Range> intersection_vec;
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

Index TensorSliceView::OffsetTo(const TensorSliceView& other) const {
  CHECK_EQ(other.NumAxes(), NumAxes());
  std::vector<int64_t> indices_vec(range_vec_.size());
  std::transform(range_vec_.cbegin(), range_vec_.cend(), other.range_vec_.cbegin(),
                 indices_vec.begin(),
                 [](const Range& lhs, const Range& rhs) { return lhs.begin() - rhs.begin(); });
  return Index(indices_vec);
}

void TensorSliceView::ToProto(TensorSliceViewProto* proto) const {
  for (const Range& range : range_vec_) { range.ToProto(proto->mutable_dim()->Add()); }
}

}  // namespace oneflow
