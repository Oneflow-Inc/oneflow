#include "oneflow/core/register/partial_tensor_view_desc.h"

namespace oneflow {

PartialTensorViewDesc::PartialTensorViewDesc(const std::initializer_list<Range>& ranges)
    : range_vec_(ranges) {
  UpdateShape();
}

PartialTensorViewDesc::PartialTensorViewDesc(const std::vector<Range>& ranges)
    : range_vec_(ranges) {
  UpdateShape();
}

PartialTensorViewDesc& PartialTensorViewDesc::operator=(const PartialTensorViewDesc& other) {
  range_vec_ = other.range_vec_;
  UpdateShape();
  return *this;
}

bool PartialTensorViewDesc::operator==(const PartialTensorViewDesc& rhs) const {
  return range_vec_ == rhs.range_vec_;
}

bool PartialTensorViewDesc::operator!=(const PartialTensorViewDesc& rhs) const {
  return !(*this == rhs);
}

void PartialTensorViewDesc::UpdateShape() {
  std::vector<int64_t> dim_vec(range_vec_.size());
  std::transform(range_vec_.cbegin(), range_vec_.cend(), dim_vec.begin(),
                 [](const Range& range) { range.size(); });
  shape_ = Shape(dim_vec);
}

bool PartialTensorViewDesc::IsEmpty() const { return range_vec_.empty(); }

PartialTensorViewDesc PartialTensorViewDesc::Intersect(const PartialTensorViewDesc& other) const {
  CHECK_EQ(other.range_vec_.size(), range_vec_.size());
  std::vector<Range> intersection_vec;
  const Range zero(0, 0);
  FOR_RANGE(int64_t, i, 0, range_vec_.size()) {
    const Range intersection = FindIntersectant(range_vec_.at(i), other.range_vec_.at(i));
    if (intersection == zero) {
      return PartialTensorViewDesc();
    } else {
      intersection_vec.emplace_back(intersection);
    }
  }
  return PartialTensorViewDesc(intersection_vec);
}

const Range& PartialTensorViewDesc::At(int64_t index) const { return range_vec_.at(index); }

const Shape& PartialTensorViewDesc::shape() const { return shape_; }

size_t PartialTensorViewDesc::size() const { return range_vec_.size(); }

void PartialTensorViewDesc::JointFold(const PartialTensorViewDesc& lhs,
                                      const PartialTensorViewDesc& rhs,
                                      PartialTensorViewDesc* lhs_out,
                                      PartialTensorViewDesc* rhs_out) {
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
  *lhs_out = PartialTensorViewDesc(lhs_out_vec);
  *rhs_out = PartialTensorViewDesc(rhs_out_vec);
}

}  // namespace oneflow
