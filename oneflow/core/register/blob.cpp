#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

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
    TensorPodDesc dense_shape_desc = header_ptr_.Field(FieldKey::kDense).pod_desc().Cast<TensorPodDesc>();
    CHECK_EQ(1, dense_shape_desc.shape().NumAxes());
    dense_shape_->Init(header_ptr_.MutTensorPtr<int64_t>(FieldKey::kDense, nullptr),
        dense_shape_desc.shape().elem_cnt());
  }

  if (header_ptr_.HasField(FieldKey::kLoD)) {
    int64_t num_of_lod_levels = blob_desc_->blob_desc_proto().num_of_lod_levels();
    CHECK_GT(num_of_lod_levels, 0);
    TensorPodDesc lod_desc = header_ptr_.Field(FieldKey::kLoD).pod_desc().Cast<TensorPodDesc>();
    CHECK_EQ(1, lod_desc.shape().NumAxes());
    lod_->Init(header_ptr_.MutTensorPtr<int64_t>(FieldKey::kLoD, nullptr), 
        lod_desc.shape().elem_cnt(), num_of_lod_levels);
  }
}

}  // namespace oneflow
