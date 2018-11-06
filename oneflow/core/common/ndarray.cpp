#include "oneflow/core/common/ndarray.h"

namespace oneflow {

template<typename T, int NDIMS>
void AssignVarNdArray(VarNdArray<T, NDIMS>* var_ndarray, const NdArray<T, NDIMS>& ndarray) {
  T* dst_ptr = nullptr;
  size_t dst_size = 0;
  T* src_ptr = nullptr;
  size_t src_size = 0;
  int64_t cur_index = 0;
  CHECK_EQ(var_ndarray->shape(), ndarray.shape());
  size_t total_elem_cnt = var_ndarray->shape().elem_cnt();
  while (cur_index < total_elem_cnt) {
    if (dst_size == 0) { var_ndarray->GetMutPtrAndContiguousSize(cur_index, &dst_ptr, &dst_size); }
    if (src_size == 0) { ndarray.GetMutPtrAndContiguousSize(cur_index, &src_ptr, &src_size); }
    if (src_size == 0) { break; }
    size_t cp_size = std::min(dst_size, src_size);
    if (cp_size == 1) {
      *dst_ptr = *src_ptr;
    } else {
      memcpy(dst_ptr, src_ptr, sizeof(T) * cp_size);
    }
    dst_ptr += cp_size;
    src_ptr += cp_size;
    dst_size -= cp_size;
    src_size -= cp_size;
    cur_index += cp_size;
  }
  CHECK_EQ(dst_size, 0);
  CHECK_EQ(src_size, 0);
  CHECK_EQ(cur_index, total_elem_cnt);
}

Slice::Slice(const std::initializer_list<int64_t>& l) {
  std::vector<int64_t> vec(l);
  value_capacity_ = 0;
  if (vec.size() == 0) {
    start_ = kStart;
    end_ = kEnd;
    stride_ = 1;
  } else if (vec.size() == 1) {
    start_ = vec[0];
    end_ = kEnd;
    stride_ = 1;
  } else if (vec.size() == 2) {
    start_ = vec[0];
    end_ = vec[1];
    stride_ = 1;
  } else if (vec.size() == 3) {
    start_ = vec[0];
    end_ = vec[1];
    stride_ = vec[2];
  } else {
    UNIMPLEMENTED();
  }
}

bool Slice::IsBounded() const {
  CHECK_NE(stride_, 0);
  if (value_capacity_ == 0) { return false; }
  return (start_ >= 0) && (start_ <= value_capacity_ - (stride_ < 0)) && (end_ >= 0 - (stride_ < 0))
         && (end_ <= value_capacity_);
}

void Slice::Bound(size_t value_capacity) {
  CHECK_GT(value_capacity, 0);
  if (value_capacity_ == value_capacity) { return; }
  CHECK_EQ(value_capacity_, 0);
  value_capacity_ = value_capacity;
  if (start_ != kStart && start_ < 0) { start_ += value_capacity_; }
  if (end_ != kStart && end_ < 0) { end_ += value_capacity_; }
  if (start_ == kStart) { start_ = 0; }
  if (end_ == kEnd) { end_ = value_capacity_; }
  if (start_ == kEnd) { start_ = value_capacity_ - (stride_ < 0); }
  if (end_ == kStart) { end_ = 0 - (stride_ < 0); }
  CHECK_NE(stride_, 0);
  CHECK_GE(start_, 0);
  CHECK_LE(start_, value_capacity_);
  CHECK_GE(end_, 0);
  CHECK_LE(end_, value_capacity_);
}

size_t Slice::Size() const {
  CHECK(IsBounded());
  if (stride_ > 0 && start_ >= end_) { return 0; }
  if (stride_ < 0 && start_ <= end_) { return 0; }
  return ((end_ - start_) + (stride_ - ((stride_ > 0) - (stride_ < 0)))) / stride_;
}

int64_t Slice::At(int64_t index) const { return start_ + index * stride_; }
bool Slice::is_contiguous() const { return start_ == 0 && end_ == value_capacity_ && stride_ == 1; }

}  // namespace oneflow
