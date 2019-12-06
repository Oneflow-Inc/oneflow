#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/register/register.h"
#include "oneflow/core/common/range.h"

namespace oneflow {

const MemoryCase& Blob::mem_case() const { return mem_case_; }

Blob::Blob(const MemoryCase& mem_case, const RtBlobDesc* blob_desc, char* header_ptr) {
  Init(mem_case, blob_desc, header_ptr, header_ptr + blob_desc->ByteSizeOfBlobHeader());
}

Blob::Blob(const MemoryCase& mem_case, const RtBlobDesc* blob_desc, char* header_ptr,
           char* body_ptr) {
  Init(mem_case, blob_desc, header_ptr, body_ptr);
}

const Shape& Blob::header_field_shape(FieldKey field_key) const {
  return blob_desc().header_pod_desc().Field(field_key).Cast<TensorPodDesc>().shape();
}

const int64_t* Blob::header_field(FieldKey field_key) const {
  return header_ptr_->TensorPtr<int64_t>(field_key);
}

int64_t* Blob::mut_header_field(FieldKey field_key) {
  return header_ptr_->MutTensorPtr<int64_t>(field_key);
}

void Blob::Init(const MemoryCase& mem_case, const RtBlobDesc* blob_desc, char* header_ptr,
                char* body_ptr) {
  mem_case_ = mem_case;
  blob_desc_ = blob_desc;
  dptr_ = body_ptr;
  header_ptr_.reset(new PodPtr(blob_desc_->header_pod_desc(), header_ptr));
  if (!blob_desc_->header_is_opaque()) {
    *mut_header_field(FieldKey::kDenseShapeListLength) = 1;
    *mut_header_field(FieldKey::kShapeListSlices) = 0;
    *mut_header_field(FieldKey::kShapeListSlicesLength) = 1;
    sole_tensor_.reset(new TensorView(header_field(FieldKey::kDenseShape), static_shape().NumAxes(),
                                      data_type(), dptr()));
    sole_mut_tensor_.reset(new DataOnlyMutTensorView(mut_header_field(FieldKey::kDenseShape),
                                                     static_shape().NumAxes(), data_type(),
                                                     mut_dptr()));
    int64_t* shape_ptr = mut_header_field(FieldKey::kDenseShape);
    dense_shape_view_.reset(new DenseShapeView(shape_ptr, static_shape().NumAxes()));
    if (blob_desc->is_dynamic()) {
      dense_shape_mut_view_.reset(new DenseShapeMutView(shape_ptr, static_shape().NumAxes()));
    }
    DenseShapeMutView(shape_ptr, static_shape().NumAxes()).set_shape(static_shape());
  } else {
    const DimVector& dim_vec = static_shape().dim_vec();
    dense_shape_view_.reset(new DenseShapeView(dim_vec.data(), dim_vec.size()));
  }
}

size_t Blob::total_num_of_tensors() const {
  size_t num_tensor = *header_field(FieldKey::kDenseShapeListLength);
  CHECK_LE(num_tensor * static_shape().NumAxes(),
           header_field_shape(FieldKey::kDenseShape).elem_cnt());
  return num_tensor;
}

const TensorView& Blob::sole_tensor() const {
  CHECK(static_cast<bool>(sole_tensor_));
  CHECK_EQ(*header_field(FieldKey::kShapeListSlicesLength), 1);
  CHECK_EQ(*header_field(FieldKey::kDenseShapeListLength), 1);
  return *sole_tensor_;
}

DataOnlyMutTensorView* Blob::sole_mut_tensor() {
  CHECK_EQ(*header_field(FieldKey::kShapeListSlicesLength), 1);
  CHECK_EQ(*header_field(FieldKey::kDenseShapeListLength), 1);
  return sole_mut_tensor_.get();
}

std::unique_ptr<TensorView> Blob::first_tensor() const {
  if (total_num_of_tensors() == 0) { return std::unique_ptr<TensorView>(); }
  return std::make_unique<TensorView>(header_field(FieldKey::kDenseShape), static_shape().NumAxes(),
                                      data_type(), dptr());
}

std::unique_ptr<TensorView> Blob::next_tensor(const TensorView& last) const {
  const int64_t* shape_ptr = last.shape_ptr() + static_shape().NumAxes();
  size_t shape_list_capacity =
      *header_field(FieldKey::kDenseShapeListLength) * static_shape().NumAxes();
  if (shape_ptr >= header_field(FieldKey::kDenseShape) + shape_list_capacity) {
    return std::unique_ptr<TensorView>();
  }
  const char* mem_ptr = reinterpret_cast<const char*>(last.dptr());
  mem_ptr += last.shape().elem_cnt() * GetSizeOfDataType(data_type());
  return std::make_unique<TensorView>(shape_ptr, static_shape().NumAxes(), data_type(),
                                      reinterpret_cast<const void*>(mem_ptr));
}

std::unique_ptr<DataOnlyMutTensorView> Blob::first_mut_tensor() {
  if (total_num_of_tensors() == 0) { return std::unique_ptr<DataOnlyMutTensorView>(); }
  return std::make_unique<DataOnlyMutTensorView>(mut_header_field(FieldKey::kDenseShape),
                                                 static_shape().NumAxes(), data_type(), mut_dptr());
}

std::unique_ptr<DataOnlyMutTensorView> Blob::next_mut_tensor(const DataOnlyMutTensorView& last) {
  const int64_t* shape_ptr = last.shape_ptr() + static_shape().NumAxes();
  size_t shape_list_capacity =
      *header_field(FieldKey::kDenseShapeListLength) * static_shape().NumAxes();
  if (shape_ptr >= header_field(FieldKey::kDenseShape) + shape_list_capacity) {
    return std::unique_ptr<DataOnlyMutTensorView>();
  }
  char* mem_ptr = reinterpret_cast<char*>(last.mut_dptr());
  mem_ptr += last.shape().elem_cnt() * GetSizeOfDataType(data_type());
  return std::make_unique<DataOnlyMutTensorView>(shape_ptr, static_shape().NumAxes(), data_type(),
                                                 reinterpret_cast<void*>(mem_ptr));
}

std::unique_ptr<FullyMutTensorView> Blob::add_tensor(const FullyMutTensorView* last) {
  if (last == nullptr) {
    CHECK_EQ(total_num_of_tensors(), 0);
  } else {
    int32_t shape_offset = (total_num_of_tensors() - 1) * static_shape().NumAxes();
    CHECK_EQ(last->shape().ptr(), header_field(FieldKey::kDenseShape) + shape_offset);
  }
  *mut_header_field(FieldKey::kDenseShapeListLength) += 1;
  if (last == nullptr) {
    return std::make_unique<FullyMutTensorView>(mut_header_field(FieldKey::kDenseShape),
                                                static_shape().NumAxes(), data_type(), mut_dptr(),
                                                blob_desc().ByteSizeOfBlobBody());
  }
  int64_t* shape_ptr = last->mut_shape_ptr() + static_shape().NumAxes();
  size_t shape_list_capacity = header_field_shape(FieldKey::kDenseShape).elem_cnt();
  const int64_t* end_shape_ptr = header_field(FieldKey::kDenseShape) + shape_list_capacity;
  CHECK_LE(shape_ptr, end_shape_ptr);
  if (shape_ptr == end_shape_ptr) { return std::unique_ptr<FullyMutTensorView>(); }
  char* mem_ptr = reinterpret_cast<char*>(last->mut_dptr());
  mem_ptr += last->shape().elem_cnt() * GetSizeOfDataType(data_type());
  char* end_ptr = reinterpret_cast<char*>(mut_dptr()) + blob_desc().ByteSizeOfBlobBody();
  CHECK_LE(mem_ptr, end_ptr);
  if (mem_ptr == end_ptr) { return std::unique_ptr<FullyMutTensorView>(); }
  return std::make_unique<FullyMutTensorView>(shape_ptr, static_shape().NumAxes(), data_type(),
                                              reinterpret_cast<void*>(mem_ptr), end_ptr - mem_ptr);
}

size_t Blob::num_of_tensor_list_slices() const {
  size_t num_slices = *header_field(FieldKey::kShapeListSlicesLength);
  CHECK_LE(num_slices, header_field_shape(FieldKey::kShapeListSlices).elem_cnt());
  return num_slices;
}

int64_t Blob::tensor_index4slice_id(int32_t slice_id) const {
  CHECK_LT(slice_id, num_of_tensor_list_slices());
  return header_field(FieldKey::kShapeListSlices)[slice_id];
}

void Blob::add_tensor_list_slice() {
  size_t slice_buff_byte_size =
      blob_desc().header_pod_desc().Field(FieldKey::kShapeListSlices).ByteSize();
  CHECK_LT(num_of_tensor_list_slices() * sizeof(int64_t), slice_buff_byte_size);
  int32_t slice_id = num_of_tensor_list_slices();
  *mut_header_field(FieldKey::kShapeListSlicesLength) += 1;
  mut_header_field(FieldKey::kShapeListSlices)[slice_id] =
      *header_field(FieldKey::kDenseShapeListLength);
}

std::unique_ptr<TensorListView> Blob::tensor_list(int32_t slice_id) const {
  Range range;
  int64_t shape_offset = 0;
  int64_t data_offset = 0;
  GetTensorListInfoBySliceId(slice_id, &range, &shape_offset, &data_offset);
  const char* mem_ptr = reinterpret_cast<const char*>(dptr());
  return std::make_unique<TensorListView>(
      range.size(), header_field(FieldKey::kDenseShape) + shape_offset, static_shape().NumAxes(),
      data_type(), reinterpret_cast<const void*>(mem_ptr + data_offset));
}

std::unique_ptr<MutTensorListView> Blob::mut_tensor_list(int32_t slice_id) {
  Range range;
  int64_t shape_offset = 0;
  int64_t data_offset = 0;
  GetTensorListInfoBySliceId(slice_id, &range, &shape_offset, &data_offset);
  char* mem_ptr = reinterpret_cast<char*>(mut_dptr());
  return std::make_unique<MutTensorListView>(
      range.size(), header_field(FieldKey::kDenseShape) + shape_offset, static_shape().NumAxes(),
      data_type(), reinterpret_cast<void*>(mem_ptr + data_offset));
}

void Blob::reserve_one_empty_tensor_list() {
  clear_tensor_lists();
  add_tensor_list_slice();
}

void Blob::clear_tensor_lists() {
  *mut_header_field(FieldKey::kDenseShapeListLength) = 0;
  *mut_header_field(FieldKey::kShapeListSlicesLength) = 0;
}

void Blob::GetTensorListSliceRange(int32_t slice_id, Range* range) const {
  CHECK_GE(slice_id, 0);
  CHECK_LT(slice_id, num_of_tensor_list_slices());
  range->mut_begin() = header_field(FieldKey::kShapeListSlices)[slice_id];
  if (slice_id == num_of_tensor_list_slices() - 1) {
    range->mut_end() = *header_field(FieldKey::kDenseShapeListLength);
  } else {
    range->mut_end() = header_field(FieldKey::kShapeListSlices)[slice_id + 1];
  }
  CHECK_LE(range->begin(), range->end());
  CHECK_LE(range->end(), *header_field(FieldKey::kDenseShapeListLength));
}

void Blob::GetTensorListInfoBySliceId(int32_t slice_id, Range* range, int64_t* shape_offset,
                                      int64_t* data_offset) const {
  GetTensorListSliceRange(slice_id, range);
  int32_t num_axes = static_shape().NumAxes();
  *data_offset = 0;
  const int64_t* shape_ptr = header_field(FieldKey::kDenseShape);
  FOR_RANGE(int32_t, i, 0, range->begin()) {
    int32_t cur_elem_cnt = 1;
    FOR_RANGE(int32_t, j, i * num_axes, (i + 1) * num_axes) { cur_elem_cnt *= shape_ptr[j]; }
    *data_offset += cur_elem_cnt * GetSizeOfDataType(data_type());
  }
  *shape_offset = range->begin() * num_axes;
}

size_t Blob::ByteSizeOfBlobBody() const {
  if (blob_desc().header_is_opaque()) { return blob_desc().ByteSizeOfBlobBody(); }
  size_t elem_cnt = 0;
  const size_t num_axes = static_shape().NumAxes();
  const int64_t* dense_shape_list = header_field(FieldKey::kDenseShape);
  FOR_RANGE(int32_t, i, 0, *header_field(FieldKey::kDenseShapeListLength)) {
    elem_cnt += DenseShapeView(dense_shape_list + i * num_axes, num_axes).elem_cnt();
  }
  size_t byte_size = elem_cnt * GetSizeOfDataType(data_type());
  CHECK_LE(byte_size, blob_desc().ByteSizeOfBlobBody());
  return byte_size;
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
    const int64_t shape_list_len = *rhs->header_field(FieldKey::kDenseShapeListLength);
    *mut_header_field(FieldKey::kDenseShapeListLength) = shape_list_len;
    const size_t num_axes = static_shape().NumAxes();
    Memcpy<DeviceType::kCPU>(device_ctx, mut_header_field(FieldKey::kDenseShape),
                             rhs->header_field(FieldKey::kDenseShape),
                             shape_list_len * num_axes * sizeof(int64_t));
  }
  {
    const int64_t seg_length = *rhs->header_field(FieldKey::kShapeListSlicesLength);
    *mut_header_field(FieldKey::kShapeListSlicesLength) = seg_length;
    Memcpy<DeviceType::kCPU>(device_ctx, mut_header_field(FieldKey::kShapeListSlices),
                             rhs->header_field(FieldKey::kShapeListSlices),
                             seg_length * sizeof(int64_t));
  }
}

bool Blob::IsBodyEmpty() const {
  const int64_t* shape_list_size = header_field(FieldKey::kDenseShapeListLength);
  if (shape_list_size == nullptr) { return false; }
  const int64_t shape_list_len = *shape_list_size;
  return shape_list_len == 0 || shape().elem_cnt() == 0;
}

}  // namespace oneflow
