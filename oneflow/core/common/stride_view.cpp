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
#include "oneflow/core/common/stride.h"
#include "oneflow/core/common/stride.pb.h"
#include "oneflow/core/common/stride_view.h"

namespace oneflow {

StrideView::StrideView(const StrideProto& stride_proto)
    : StrideViewBase<const int64_t>(stride_proto.dim().data(), stride_proto.dim_size()) {}
StrideView::StrideView(const Stride& stride)
    : StrideViewBase<const int64_t>(stride.StrideVec().data(), stride.StrideVec().size()) {}

template<typename DimT>
int64_t StrideViewBase<DimT>::At(int64_t index) const {
  CHECK_GE(index, 0);
  if (!(this->NumAxes() == 0 && this->ElemCnt() == 1)) {
    CHECK_LT(index, num_axes_);
  } else {
    CHECK(index == 0);
  }
  return ptr_[index];
}

template<typename DimT>
int64_t StrideViewBase<DimT>::Count(int64_t begin_axis) const {
  return this->Count(begin_axis, NumAxes());
}

template<typename DimT>
int64_t StrideViewBase<DimT>::Count(int64_t begin_axis, int64_t end_axis) const {
  CHECK(0 <= begin_axis && begin_axis <= end_axis && end_axis <= this->NumAxes())
      << begin_axis << " " << end_axis;
  int64_t cnt = 1;
  for (int64_t i = begin_axis; i < end_axis; ++i) { cnt *= this->At(i); }
  return cnt;
}

template<typename DimT>
int64_t StrideViewBase<DimT>::ElemCnt() const {
  return this->Count(0);
}

template<typename DimT>
std::string StrideViewBase<DimT>::ToString() const {
  std::stringstream ss;
  ss << "(";
  FOR_RANGE(int, i, 0, this->NumAxes()) {
    int64_t dim = this->At(i);
    ss << dim;
    if (i != this->NumAxes() - 1 || this->NumAxes() == 1) { ss << ","; }
  }
  ss << ")";
  return ss.str();
}

template<typename DimT>
void StrideViewBase<DimT>::ToStrideVector(StrideVector* stride_vec) const {
  stride_vec->resize(num_axes_);
  stride_vec->assign(ptr_, ptr_ + num_axes_);
}

template<typename DimT>
void StrideViewBase<DimT>::ToStride(Stride* stride) const {
  StrideVector stride_vec;
  this->ToStrideVector(&stride_vec);
  stride->assign(stride_vec);
}

template class StrideViewBase<const int64_t>;
template class StrideViewBase<int64_t>;

std::ostream& operator<<(std::ostream& out, const StrideView& stride) {
  out << stride.ToString();
  return out;
}

void MutStrideView::Set(int64_t axis, int64_t val) {
  CHECK_GE(axis, 0);
  CHECK_LT(axis, NumAxes());
  dim_ptr()[axis] = val;
}

void MutStrideView::set_stride(const Stride& stride) {
  CHECK_EQ(NumAxes(), stride.NumAxes());
  std::copy(stride.StrideVec().data(), stride.StrideVec().data() + stride.NumAxes(), dim_ptr());
}

void MutStrideView::set_stride(const StrideView& stride) {
  CHECK_EQ(NumAxes(), stride.NumAxes());
  std::copy(stride.ptr(), stride.ptr() + stride.NumAxes(), dim_ptr());
}

}  // namespace oneflow
