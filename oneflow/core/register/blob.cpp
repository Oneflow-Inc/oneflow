#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

DenseShapeViewBase::DenseShapeViewBase(PodPtr dense_shape_ptr) {
  ptr_ = dense_shape_ptr.MutTensorPtr<int64_t>();
  CHECK_NOTNULL(ptr_);
  const TensorPodDesc& dense_shape_desc = dense_shape_ptr.pod_desc().Cast<TensorPodDesc>();
  CHECK_EQ(1, dense_shape_desc.shape().NumAxes());
  num_axes_ = dense_shape_desc.shape().At(0);
}

DenseShapeViewBase::DenseShapeViewBase& operator=(const DenseShapeViewBase& rhs) {
  ptr_ = rhs.ptr_;
  num_axes_ = rhs.num_axes_;
  return *this;
}

int64_t DenseShapeView::At(int64_t index) const {
  CHECL_GT(index, 0);
  CHECK_LT(index, num_axes_);
  return *(ptr + index);
}

void DenseShapeMutView::set_shape(const Shape& val) {
  CHECK_EQ(num_axes_, val.NumAxes());
  shape_ = val;
  is_shape_inited_ = true;
  for (size_t i = 0; i < shape_.NumAxes(); ++i) { *(ptr_ + i) = shape_.At(i); }
}

LoDViewBase::LoDViewBase(PodPtr lod_ptr, int64_t num_of_lod_levels) {
  ptr_ = lod_ptr_.MutTensorPtr<int64_t>();
  CHECK_NOTNULL(ptr_);
  num_of_lod_levels_ = num_of_lod_levels;
  const TensorPodDesc& lod_desc = lod_ptr.pod_desc().Cast<TensorPodDesc>();
  CHECK_EQ(1, lod_desc.shape().NumAxes());
  max_reserved_size_for_lod_ = lod_desc.shape().At(0);
}

LoDViewBase& LoDViewBase::operator=(const LoDViewBase& rhs) {
  ptr_ = rhs.ptr_;
  num_of_lod_levels_ = rhs.num_of_lod_levels_;
  max_reserved_size_for_lod_ = rhs.max_reserved_size_for_lod_;
}

LoDVec LoDViewBase::InitOffsetVecFromPtr() {
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
    if ((cur_lod_level == num_of_lod_levels_ - 1) 
        && (cur_lod_level_cnt == cur_lod_level_max_cnt)) {
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

LoDVec LoDViewBase::GetLengthLoDVecFromOffsetLoDVec(
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

LoDVec LoDViewBase::GetOffsetLoDVecFromLengthLoDVec(
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
    length_lod_vec_ = GetLengthLoDVecFromOffsetLoDVec(InitLengthVecFromPtr());
  }
  return length_lod_vec_.at(level).at(pos);
}

void LengthLoDMutView::SetLength(const LoDVec& length_lod_vec) {
  LoDViewBase::FlushOffsetVecToPtr(GetOffsetLoDVecFromLengthLoDVec(length_lod_vec));
}

const MemoryCase& Blob::mem_case() const { return regst_->regst_desc()->mem_case(); }

Blob::Blob(Regst* regst, const RtBlobDesc* blob_desc, char* header_ptr) {
  Init(regst, blob_desc, header_ptr, header_ptr + blob_desc->RealByteSizeOfBlobHeader());
}

Blob::Blob(Regst* regst, const RtBlobDesc* blob_desc, char* header_ptr, char* body_ptr) {
  Init(regst, blob_desc, header_ptr, body_ptr);
}

void Blob::Init(Regst* regst, const RtBlobDesc* blob_desc, char* header_ptr, char* body_ptr) {
  is_contiguous_ = (body_ptr == header_ptr + blob_desc->RealByteSizeOfBlobHeader());
  regst_ = regst;
  blob_desc_ = blob_desc;
  dptr_ = body_ptr;
  header_ptr_ = PodPtr(blob_desc_->header_pod_desc(), header_ptr);

  {
    TensorPodDesc dense_shape_desc =
        header_ptr_.Field(FieldKey::kDenseShap).pod_desc().Cast<TensorPodDesc>();
    CHECK_EQ(1, dense_shape_desc.shape().NumAxes());
    dense_shape_.Init(header_ptr_.MutTensorPtr<int64_t>(FieldKey::kDenseShap, nullptr),
                      dense_shape_desc.shape().elem_cnt());
    dense_shape_.set_shape(blob_desc_->body_shape());
  }

  if (header_ptr_.HasField(FieldKey::kLoD)) {
    int64_t num_of_lod_levels = blob_desc_->blob_desc_proto().num_of_lod_levels();
    CHECK_GT(num_of_lod_levels, 0);
    TensorPodDesc lod_desc = header_ptr_.Field(FieldKey::kLoD).pod_desc().Cast<TensorPodDesc>();
    CHECK_EQ(1, lod_desc.shape().NumAxes());
    lod_.Init(header_ptr_.MutTensorPtr<int64_t>(FieldKey::kLoD, nullptr),
              lod_desc.shape().elem_cnt(), num_of_lod_levels);
  }
}

}  // namespace oneflow
