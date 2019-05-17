#include "oneflow/core/register/partial_tensor_view.h"

namespace oneflow {

PartialTensorView::PartialTensorView(const std::initializer_list<Range>& ranges)
    : range_vec_(ranges) {
  UpdateShape();
}

PartialTensorView::PartialTensorView(const std::vector<Range>& ranges) : range_vec_(ranges) {
  UpdateShape();
}

PartialTensorView& PartialTensorView::operator=(const PartialTensorView& other) {
  range_vec_ = other.range_vec_;
  UpdateShape();
  return *this;
}

bool PartialTensorView::operator==(const PartialTensorView& rhs) const {
  return range_vec_ == rhs.range_vec_;
}

bool PartialTensorView::operator!=(const PartialTensorView& rhs) const { return !(*this == rhs); }

void PartialTensorView::UpdateShape() {
  std::vector<int64_t> dim_vec(range_vec_.size());
  std::transform(range_vec_.cbegin(), range_vec_.cend(), dim_vec.begin(),
                 [](const Range& range) { return range.size(); });
  shape_ = Shape(dim_vec);
}

bool PartialTensorView::IsEmpty() const { return range_vec_.empty(); }

PartialTensorView PartialTensorView::Intersect(const PartialTensorView& other) const {
  CHECK_EQ(other.range_vec_.size(), range_vec_.size());
  std::vector<Range> intersection_vec;
  const Range zero(0, 0);
  FOR_RANGE(int64_t, i, 0, range_vec_.size()) {
    const Range intersection = FindIntersectant(range_vec_.at(i), other.range_vec_.at(i));
    if (intersection == zero) {
      return PartialTensorView();
    } else {
      intersection_vec.emplace_back(intersection);
    }
  }
  return PartialTensorView(intersection_vec);
}

const Range& PartialTensorView::At(int64_t index) const { return range_vec_.at(index); }

const Shape& PartialTensorView::shape() const { return shape_; }

const std::vector<Range>& PartialTensorView::range_vec() const { return range_vec_; }

size_t PartialTensorView::NumAxes() const { return range_vec_.size(); }

Index PartialTensorView::OffsetTo(const PartialTensorView& other) {
  CHECK_EQ(other.NumAxes(), NumAxes());
  std::vector<int64_t> indices_vec;
  std::transform(range_vec_.cbegin(), range_vec_.cend(), other.range_vec_.cbegin(),
                 [](const Range& lhs, const Range& rhs) { return rhs.begin() - lhs.begin(); });
  return Index(indices_vec);
}

void PartialTensorView::FoldSameRange(const PartialTensorView& lhs, const PartialTensorView& rhs,
                                      PartialTensorView* lhs_out, PartialTensorView* rhs_out) {
  std::vector<Range> lhs_out_vec;
  std::vector<Range> rhs_out_vec;
  CHECK_EQ(lhs.range_vec_.size(), rhs.range_vec_.size());
  FOR_RANGE(int64_t, i, 0, lhs.range_vec_.size()) {
    const Range& lhs_range = lhs.range_vec_.at(i);
    const Range& rhs_range = rhs.range_vec_.at(i);
    if (lhs_range == rhs_range) {
      if (lhs_out_vec.empty()) {
        CHECK(rhs_out_vec.empty());
        lhs_out_vec.push_back(lhs_range);
        rhs_out_vec.push_back(lhs_range);
      } else {
        lhs_out_vec.back().mut_begin() *= lhs_range.size();
        lhs_out_vec.back().mut_end() *= lhs_range.size();
      }
    } else {
      lhs_out_vec.push_back(lhs_range);
      rhs_out_vec.push_back(rhs_range);
    }
  }
  *lhs_out = PartialTensorView(lhs_out_vec);
  *rhs_out = PartialTensorView(rhs_out_vec);
}

}  // namespace oneflow
