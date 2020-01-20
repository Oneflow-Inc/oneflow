#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

TensorBackInserter::TensorBackInserter(Blob* blob)
    : blob_(blob), cur_mut_tensor_(blob->EndFullyMutTensor()) {}

void TensorBackInserter::ReserveOneEmptyTensorList() {
  blob_->ReserveOneEmptyTensorList();
  if (IsCurMutTensorAvailable()) { cur_mut_tensor_ = blob_->EndFullyMutTensor(); }
}

void TensorBackInserter::ClearTensorLists() {
  blob_->clear_tensor_lists();
  if (IsCurMutTensorAvailable()) { cur_mut_tensor_ = blob_->EndFullyMutTensor(); }
}

bool TensorBackInserter::IsCurMutTensorAvailable() const {
  return !blob_->IsEndFullyMutTensor(cur_mut_tensor_);
}

FullyMutTensorView* TensorBackInserter::add_tensor() {
  blob_->AddTensor(&cur_mut_tensor_);
  return &cur_mut_tensor_;
}

FullyMutTensorView* TensorBackInserter::cur_mut_tensor() { return &cur_mut_tensor_; }

void TensorBackInserter::add_tensor_list_slice() { return blob_->add_tensor_list_slice(); }

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
  mem_case_ = mem_case;
  blob_desc_ = blob_desc;
  dptr_ = body_ptr;
  header_ptr_.reset(new PodPtr(blob_desc_->header_pod_desc(), header_ptr));
  FOR_RANGE(int32_t, i, 0, FieldKey::kFieldKeySize) {
    FieldKey key = static_cast<FieldKey>(i);
    header_fields_[i] = header_ptr_->MutTensorPtr<int64_t>(key);
    if (header_fields_[i] == nullptr) {
      header_field_capacities_[i] = 0;
    } else {
      header_field_capacities_[i] =
          blob_desc->header_pod_desc().Field(key).Cast<TensorPodDesc>().shape().elem_cnt();
    }
  }
  if (!blob_desc_->header_is_opaque()) {
    *mut_header_field<FieldKey::kTensorListLength>() = 1;
    *mut_header_field<FieldKey::kTensorListSlices>() = 0;
    *mut_header_field<FieldKey::kTensorListSlicesLength>() = 1;
    *mut_header_field<FieldKey::kLastTensorDataOffset>() = 0;
    begin_tensor_.reset(
        new TensorView(this, header_field<FieldKey::kTensorShapeList>(), dptr<char>()));
    begin_mut_tensor_.reset(new DataOnlyMutTensorView(
        this, mut_header_field<FieldKey::kTensorShapeList>(), mut_dptr<char>()));
    int64_t* shape_ptr = mut_header_field<FieldKey::kTensorShapeList>();
    shape_view_.reset(new ShapeView(shape_ptr, static_shape().NumAxes()));
    if (blob_desc->is_dynamic()) {
      mut_shape_view_.reset(new MutShapeView(shape_ptr, static_shape().NumAxes()));
    }
    MutShapeView(shape_ptr, static_shape().NumAxes()).set_shape(static_shape());
  } else {
    const DimVector& dim_vec = static_shape().dim_vec();
    shape_view_.reset(new ShapeView(dim_vec.data(), dim_vec.size()));
  }
}

size_t Blob::total_num_of_tensors() const {
  size_t num_tensor = *header_field<FieldKey::kTensorListLength>();
  CHECK_LE(num_tensor * static_shape().NumAxes(),
           header_field_capacity<FieldKey::kTensorShapeList>());
  return num_tensor;
}

void Blob::MoveToNextTensor(TensorView* last) const {
  CHECK(!IsEndTensor(*last));
  const int64_t* shape_ptr = last->shape().ptr() + static_shape().NumAxes();
  const char* dptr = last->dptr<char>() + last->ByteSize();
  last->reset(shape_ptr, dptr);
}

void Blob::MoveToNextMutTensor(DataOnlyMutTensorView* last) {
  CHECK(!IsEndTensor(*last));
  const int64_t* shape_ptr = last->shape().ptr() + static_shape().NumAxes();
  char* dptr = last->mut_dptr<char>() + last->ByteSize();
  last->reset(shape_ptr, dptr);
}

bool Blob::IsEndTensor(const TensorView& tensor) const {
  const int64_t* end_shape_ptr =
      header_field<FieldKey::kTensorShapeList>()
      + *header_field<FieldKey::kTensorListLength>() * static_shape().NumAxes();
  return end_shape_ptr == tensor.shape().ptr();
}

bool Blob::IsEndTensor(const DataOnlyMutTensorView& tensor) const {
  const int64_t* end_shape_ptr =
      header_field<FieldKey::kTensorShapeList>()
      + *header_field<FieldKey::kTensorListLength>() * static_shape().NumAxes();
  return end_shape_ptr == tensor.shape().ptr();
}

bool Blob::IsEndFullyMutTensor(const FullyMutTensorView& tensor) const {
  size_t shape_list_capacity = header_field_capacity<FieldKey::kTensorShapeList>();
  const int64_t* end_shape_ptr = header_field<FieldKey::kTensorShapeList>() + shape_list_capacity;
  const char* end_tensor_dptr = dptr<char>() + blob_desc().ByteSizeOfBlobBody();
  return tensor.shape().ptr() == end_shape_ptr || tensor.dptr<char>() == end_tensor_dptr;
}

const TensorView& Blob::sole_tensor() const {
  CHECK(static_cast<bool>(begin_tensor_));
  CHECK_EQ(*header_field<FieldKey::kTensorListSlicesLength>(), 1);
  CHECK_EQ(*header_field<FieldKey::kTensorListLength>(), 1);
  CHECK_EQ(*header_field<FieldKey::kLastTensorDataOffset>(), 0);
  return *begin_tensor_;
}

const DataOnlyMutTensorView& Blob::sole_mut_tensor() {
  CHECK(static_cast<bool>(begin_mut_tensor_));
  CHECK_EQ(*header_field<FieldKey::kTensorListSlicesLength>(), 1);
  CHECK_EQ(*header_field<FieldKey::kTensorListLength>(), 1);
  CHECK_EQ(*header_field<FieldKey::kLastTensorDataOffset>(), 0);
  return *begin_mut_tensor_;
}

void Blob::AddTensor(FullyMutTensorView* tensor) {
  size_t end_tensor_data_offset = GetEndTensorDataOffset();
  int64_t* shape_ptr = mut_header_field<FieldKey::kTensorShapeList>();
  shape_ptr += total_num_of_tensors() * static_shape().NumAxes();
  tensor->reset(shape_ptr, mut_dptr<char>() + end_tensor_data_offset);
  *mut_header_field<FieldKey::kTensorListLength>() += 1;
  if (end_tensor_data_offset == 0) {
    CHECK_EQ(total_num_of_tensors(), 1);
    CHECK_EQ(*header_field<FieldKey::kLastTensorDataOffset>(), 0);
  } else {
    *mut_header_field<FieldKey::kLastTensorDataOffset>() = end_tensor_data_offset;
  }
}

size_t Blob::num_of_tensor_list_slices() const {
  size_t num_slices = *header_field<FieldKey::kTensorListSlicesLength>();
  CHECK_LE(num_slices, header_field_capacity<FieldKey::kTensorListSlices>());
  return num_slices;
}

int64_t Blob::tensor_index4slice_id(int32_t slice_id) const {
  CHECK_LT(slice_id, num_of_tensor_list_slices());
  return header_field<FieldKey::kTensorListSlices>()[slice_id];
}

void Blob::add_tensor_list_slice() {
  size_t slice_buff_byte_size =
      blob_desc().header_pod_desc().Field(FieldKey::kTensorListSlices).ByteSize();
  CHECK_LT(num_of_tensor_list_slices() * sizeof(int64_t), slice_buff_byte_size);
  int32_t slice_id = num_of_tensor_list_slices();
  *mut_header_field<FieldKey::kTensorListSlicesLength>() += 1;
  mut_header_field<FieldKey::kTensorListSlices>()[slice_id] =
      *header_field<FieldKey::kTensorListLength>();
}

void Blob::ReserveOneEmptyTensorList() {
  clear_tensor_lists();
  add_tensor_list_slice();
}

FullyMutTensorView Blob::EndFullyMutTensor() {
  size_t shape_list_capacity = header_field_capacity<FieldKey::kTensorShapeList>();
  int64_t* end_shape_ptr = mut_header_field<FieldKey::kTensorShapeList>() + shape_list_capacity;
  char* end_tensor_dptr = mut_dptr<char>() + blob_desc().ByteSizeOfBlobBody();
  return FullyMutTensorView(this, end_shape_ptr, end_tensor_dptr);
}

void Blob::clear_tensor_lists() {
  *mut_header_field<FieldKey::kTensorListLength>() = 0;
  *mut_header_field<FieldKey::kTensorListSlicesLength>() = 0;
  *mut_header_field<FieldKey::kLastTensorDataOffset>() = 0;
}

size_t Blob::GetEndTensorDataOffset() const {
  size_t num_tensor = total_num_of_tensors();
  if (num_tensor == 0) { return 0; }
  const int64_t* shape_ptr = header_field<FieldKey::kTensorShapeList>();
  shape_ptr += (num_tensor - 1) * static_shape().NumAxes();
  size_t elem_cnt = 1;
  FOR_RANGE(int32_t, i, 0, static_shape().NumAxes()) { elem_cnt *= shape_ptr[i]; }
  size_t last_tensor_byte_size = elem_cnt * GetSizeOfDataType(data_type());
  return *header_field<FieldKey::kLastTensorDataOffset>() + last_tensor_byte_size;
}

size_t Blob::ByteSizeOfBlobBody() const {
  if (blob_desc().header_is_opaque()) { return blob_desc().ByteSizeOfBlobBody(); }
  return GetEndTensorDataOffset();
}

void Blob::CopyDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs) { return; }
  AutoMemcpy(device_ctx, mut_dptr(), rhs->dptr(), ByteSizeOfBlobBody(), mem_case(),
             rhs->mem_case());
}

void Blob::CopyValidDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs) { return; }
  const size_t body_byte_size = ByteSizeOfBlobBody();
  CHECK_EQ(rhs->ByteSizeOfBlobBody(), body_byte_size);
  AutoMemcpy(device_ctx, mut_dptr(), rhs->dptr(), body_byte_size, mem_case(), rhs->mem_case());
}

void Blob::CopyHeaderFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  if (this == rhs || blob_desc().ByteSizeOfBlobHeader() == 0) { return; }
  CHECK_EQ(blob_desc().ByteSizeOfBlobHeader(), rhs->blob_desc().ByteSizeOfBlobHeader());
  if (blob_desc().header_is_opaque()) {
    Memcpy<DeviceType::kCPU>(device_ctx, header_ptr_->ptr(), rhs->header_ptr(),
                             blob_desc().ByteSizeOfBlobHeader());
    return;
  }
  {
    const int64_t shape_list_len = *rhs->header_field<FieldKey::kTensorListLength>();
    *mut_header_field<FieldKey::kTensorListLength>() = shape_list_len;
    *mut_header_field<FieldKey::kLastTensorDataOffset>() =
        *rhs->header_field<FieldKey::kLastTensorDataOffset>();
    const size_t num_axes = static_shape().NumAxes();
    Memcpy<DeviceType::kCPU>(device_ctx, mut_header_field<FieldKey::kTensorShapeList>(),
                             rhs->header_field<FieldKey::kTensorShapeList>(),
                             shape_list_len * num_axes * sizeof(int64_t));
  }
  {
    const int64_t seg_length = *rhs->header_field<FieldKey::kTensorListSlicesLength>();
    *mut_header_field<FieldKey::kTensorListSlicesLength>() = seg_length;
    Memcpy<DeviceType::kCPU>(device_ctx, mut_header_field<FieldKey::kTensorListSlices>(),
                             rhs->header_field<FieldKey::kTensorListSlices>(),
                             seg_length * sizeof(int64_t));
  }
}

bool Blob::IsBodyEmpty() const {
  const int64_t* shape_list_size = header_field<FieldKey::kTensorListLength>();
  if (shape_list_size == nullptr) { return false; }
  const int64_t shape_list_len = *shape_list_size;
  return shape_list_len == 0 || shape().elem_cnt() == 0;
}

char* Blob::mut_contiguous_header_ptr() {
  // check header and body is continuous
  CHECK_EQ(header_ptr() + blob_desc_->ByteSizeOfBlobHeader(), dptr<char>());
  return header_ptr_->ptr();
}

}  // namespace oneflow
