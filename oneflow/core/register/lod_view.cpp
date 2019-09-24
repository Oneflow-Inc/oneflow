#include "oneflow/core/register/lod_view.h"

namespace oneflow {

LoDViewBase::LoDViewBase(PodPtr lod_ptr, int64_t num_of_lod_levels) {
  ptr_ = lod_ptr.MutTensorPtr<int64_t>();
  CHECK_NOTNULL(ptr_);
  num_of_lod_levels_ = num_of_lod_levels;
  const TensorPodDesc& lod_desc = lod_ptr.pod_desc().Cast<TensorPodDesc>();
  CHECK_EQ(1, lod_desc.shape().NumAxes());
  max_reserved_size_for_lod_ = lod_desc.shape().At(0);
}

LoDViewBase::LoDVec LoDViewBase::InitOffsetVecFromPtr() const {
  LoDVec offset_vec;
  offset_vec.resize(num_of_lod_levels_);
  size_t cur_lod_level = 0;
  size_t cur_lod_level_max_cnt = 0;
  size_t cur_lod_level_cnt = 0;
  int64_t* cur_pos = ptr_;

  CHECK_EQ(0, *cur_pos);
  offset_vec.at(cur_lod_level).push_back(*cur_pos);
  offset_vec.at(cur_lod_level).push_back(*(cur_pos + 1));
  cur_pos += 2;
  CHECK_EQ(0, *cur_pos);

  while (cur_lod_level < num_of_lod_levels_) {
    if ((cur_lod_level == num_of_lod_levels_ - 1) && (cur_lod_level_cnt == cur_lod_level_max_cnt)) {
      break;
    }
    if (*cur_pos == 0) {
      CHECK_EQ(cur_lod_level_max_cnt, cur_lod_level_cnt);
      cur_lod_level += 1;
      cur_lod_level_max_cnt = (*(cur_pos - 1)) + 1;
      cur_lod_level_cnt = 0;
    }
    offset_vec.at(cur_lod_level).push_back(*cur_pos);
    cur_pos += 1;
    cur_lod_level_cnt += 1;
  }
  return offset_vec;
}

void LoDViewBase::FlushOffsetVecToPtr(const LoDVec& offset_lod_vec) {
  CHECK_EQ(num_of_lod_levels_, offset_lod_vec.size());
  size_t vec_cnt = 0;
  int64_t* cur_pos = ptr_;
  for (const auto& vec : offset_lod_vec) {
    for (int64_t offset : vec) {
      *cur_pos = offset;
      cur_pos += 1;
      vec_cnt += 1;
    }
  }
  CHECK_LT(vec_cnt, max_reserved_size_for_lod_);
}

LoDViewBase::LoDVec LoDViewBase::GetLengthLoDVecFromOffsetLoDVec(
    const LoDVec& offset_lod_vec) const {
  LoDVec length_lod_vec(offset_lod_vec.size());
  for (size_t i = 0; i < offset_lod_vec.size(); ++i) {
    const std::vector<int64_t>& vec = offset_lod_vec.at(i);
    CHECK_EQ(0, vec.front());
    for (size_t j = 1; j < vec.size(); ++j) {
      length_lod_vec.at(i).push_back(vec.at(j) - vec.at(j - 1));
    }
  }
  return length_lod_vec;
}

LoDViewBase::LoDVec LoDViewBase::GetOffsetLoDVecFromLengthLoDVec(
    const LoDVec& length_lod_vec) const {
  LoDVec offset_lod_vec(length_lod_vec.size());
  for (size_t i = 0; i < length_lod_vec.size(); ++i) {
    const std::vector<int64_t>& vec = length_lod_vec.at(i);
    offset_lod_vec.at(i).push_back(0);
    for (size_t j = 0; j < vec.size(); ++j) {
      offset_lod_vec.at(i).push_back(offset_lod_vec.at(i).back() + vec.at(j));
    }
  }
  return offset_lod_vec;
}

int64_t OffsetLoDView::GetOffset(size_t level, size_t pos) {
  if (offset_lod_vec_.empty()) { offset_lod_vec_ = LoDViewBase::InitOffsetVecFromPtr(); }
  return offset_lod_vec_.at(level).at(pos);
}

void OffsetLoDMutView::SetOffset(const LoDVec& offset_lod_vec) {
  LoDViewBase::FlushOffsetVecToPtr(offset_lod_vec);
}

int64_t LengthLoDView::GetLength(size_t level, size_t pos) {
  if (length_lod_vec_.empty()) {
    length_lod_vec_ = GetLengthLoDVecFromOffsetLoDVec(InitOffsetVecFromPtr());
  }
  return length_lod_vec_.at(level).at(pos);
}

void LengthLoDMutView::SetLength(const LoDVec& length_lod_vec) {
  LoDViewBase::FlushOffsetVecToPtr(GetOffsetLoDVecFromLengthLoDVec(length_lod_vec));
}

}  // namespace oneflow
