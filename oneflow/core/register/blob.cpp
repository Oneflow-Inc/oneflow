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

DenseShapeView::operator Shape() const {
  std::vector<int64_t> dim_vec;
  FOR_RANGE(int, i, 0, NumAxes()) { dim_vec.push_back(At(i)); }
  return Shape(dim_vec);
}

bool DenseShapeView::operator==(const DenseShapeView& rhs) const {
  if (NumAxes() != rhs.NumAxes()) { return false; }
  FOR_RANGE(int, i, 0, NumAxes()) {
    if (At(i) != rhs.At(i)) { return false; }
  }
  return true;
}

int64_t DenseShapeView::At(int64_t index) const {
  CHECK_GT(index, 0);
  CHECK_LT(index, num_axes_);
  return ptr_[index];
}

int64_t DenseShapeView::Count(int64_t begin_axis) const { return Count(begin_axis, NumAxes()); }

int64_t DenseShapeView::Count(int64_t begin_axis, int64_t end_axis) const {
  CHECK(0 <= begin_axis && begin_axis <= end_axis && end_axis <= NumAxes())
      << begin_axis << " " << end_axis;
  int64_t cnt = 1;
  for (int64_t i = begin_axis; i < end_axis; ++i) { cnt *= At(i); }
  return cnt;
}

int64_t DenseShapeView::elem_cnt() const { return Count(0); }

std::string DenseShapeView::ToString() const {
  std::stringstream ss;
  ss << "(";
  FOR_RANGE(int, i, 0, NumAxes()) {
    int64_t dim = At(i);
    ss << dim;
    if (i != NumAxes() - 1 || NumAxes() == 1) { ss << ","; }
  }
  ss << ")";
  return ss.str();
}

std::ostream& operator<<(std::ostream& out, const DenseShapeView& shape) {
  out << shape.ToString();
  return out;
}

void DenseShapeMutView::set_shape(const Shape& shape) {
  CHECK_EQ(num_axes_, shape.NumAxes());
  for (size_t i = 0; i < shape.NumAxes(); ++i) { ptr_[i] = shape.At(i); }
}

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

LoDViewBase::LoDVec LoDViewBase::InitLengthVecFromPtr() const { TODO(); }

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
    length_lod_vec_ = GetLengthLoDVecFromOffsetLoDVec(InitLengthVecFromPtr());
  }
  return length_lod_vec_.at(level).at(pos);
}

void LengthLoDMutView::SetLength(const LoDVec& length_lod_vec) {
  LoDViewBase::FlushOffsetVecToPtr(GetOffsetLoDVecFromLengthLoDVec(length_lod_vec));
}

const MemoryCase& Blob::mem_case() const { return mem_case_; }

Blob::Blob(const MemoryCase& mem_case, const RtBlobDesc* blob_desc, char* header_ptr) {
  Init(mem_case, blob_desc, header_ptr, header_ptr + blob_desc->ByteSizeOfBlobHeader());
}

Blob::Blob(const MemoryCase& mem_case, const RtBlobDesc* blob_desc, char* header_ptr,
           char* body_ptr) {
  Init(mem_case, blob_desc, header_ptr, body_ptr);
}

void Blob::Init(const MemoryCase& mem_case, const RtBlobDesc* blob_desc, char* header_ptr,
                char* body_ptr) {
  is_header_body_contiguous_ = (body_ptr == header_ptr + blob_desc->ByteSizeOfBlobHeader());
  mem_case_ = mem_case;
  blob_desc_ = blob_desc;
  dptr_ = body_ptr;
  header_ptr_.reset(new PodPtr(blob_desc_->header_pod_desc(), header_ptr));
}

void Blob::CopyDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs) { return; }
  AutoMemcpy(device_ctx, mut_dptr(), rhs->dptr(), blob_desc_->ByteSizeOfBlobBody(), mem_case(),
             rhs->mem_case());
}

void Blob::CopyValidDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs) { return; }
  CHECK_EQ(rhs->shape().elem_cnt(), shape().elem_cnt());
  AutoMemcpy(device_ctx, mut_dptr(), rhs->dptr(), blob_desc_->ByteSizeOfBlobBody(), mem_case(),
             rhs->mem_case());
}

void Blob::CopyHeaderFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs || blob_desc().ByteSizeOfBlobHeader() == 0) { return; }
  CHECK_EQ(blob_desc().ByteSizeOfBlobHeader(), rhs->blob_desc().ByteSizeOfBlobHeader());
  Memcpy<DeviceType::kCPU>(device_ctx, mut_header_ptr(), rhs->header_ptr(),
                           blob_desc().ByteSizeOfBlobHeader());
}

}  // namespace oneflow
