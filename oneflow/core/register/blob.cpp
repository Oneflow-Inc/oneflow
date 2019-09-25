#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

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

void Blob::CopyDenseShapeFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  const int64_t dense_shape_byte_size =
      blob_desc().header_pod_desc().Field(FieldKey::kDenseShape).ByteSize();
  if (this == rhs || dense_shape_byte_size == 0) { return; }
  const int64_t rhs_dense_shape_byte_size =
      rhs->blob_desc().header_pod_desc().Field(FieldKey::kDenseShape).ByteSize();
  CHECK_EQ_OR_RETURN(dense_shape_byte_size, rhs_dense_shape_byte_size);
  Memcpy<DeviceType::kCPU>(
      device_ctx, rhs->mut_header_ptr()->MutTensorPtr<char>(FieldKey::kDenseShape, nullptr),
      dense_shape_byte_size);
}

void Blob::CopyLoDFrom(DeviceCtx* device_ctx, const Blob* rhs) {
  const int64_t lod_byte_size = blob_desc().header_pod_desc().Field(FieldKey::kLoD).ByteSize();
  if (this == rhs || lod_byte_size == 0) { return; }
  const int64_t rhs_lod_byte_size =
      rhs->blob_desc().header_pod_desc().Field(FieldKey::kLoD).ByteSize();
  CHECK_EQ_OR_RETURN(lod_byte_size, rhs_lod_byte_size);
  Memcpy<DeviceType::kCPU>(device_ctx,
                           rhs->mut_header_ptr()->MutTensorPtr<char>(FieldKey::kLoD, nullptr),
                           lod_byte_size);
}

}  // namespace oneflow
