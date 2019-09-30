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
  if (!blob_desc_->header_is_opaque()) {
    std::vector<int64_t> dim_vec = static_shape().dim_vec();
    if (blob_desc->num_of_lod_levels() > 0) {
      CHECK_GT(blob_desc->num_of_lod_levels(), 1);
      int64_t dim0 = 1;
      FOR_RANGE(int64_t, i, 0, blob_desc->num_of_lod_levels()) { dim0 *= dim_vec.at(i); }
      dim_vec = {dim_vec.begin() + blob_desc->num_of_lod_levels() - 1, dim_vec.end()};
    }
    dense_shape_mut_view().set_shape(Shape(dim_vec));
  }
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
