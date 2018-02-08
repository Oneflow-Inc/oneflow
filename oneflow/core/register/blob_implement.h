#ifndef ONEFLOW_CORE_REGISTER_BLOB_IMPLEMENT_H_
#define ONEFLOW_CORE_REGISTER_BLOB_IMPLEMENT_H_

#include "oneflow/core/eigen/tensor_type.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

template<typename T, int NDIMS>
class BlobImpl<T, NDIMS> : Blob {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BlobImpl);
  BlobImpl(Regst* regst, const BlobDesc* blob_desc, char* mem_ptr)
      : Blob(regst, blob_desc, mem_ptr, nullptr) {}
  BlobImpl(Regst* regst, const BlobDesc* blob_desc, char* mem_ptr,
           const void* comm_net_token);
  ~BlobImpl() = default;

  void Transpose(Blob* out_blob, std::vector<int32_t> permutation) override {}

 private:
  Tensor<T, NDIMS> tensor_;
  ConstTensor<T, NDIMS> const_tensor_;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_IMPLEMENT_H_
