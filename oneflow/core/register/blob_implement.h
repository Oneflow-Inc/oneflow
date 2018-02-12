#ifndef ONEFLOW_CORE_REGISTER_BLOB_IMPLEMENT_H_
#define ONEFLOW_CORE_REGISTER_BLOB_IMPLEMENT_H_

#include "oneflow/core/eigen/tensor_type.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

template<typename T, int32_t NDIMS, DeviceType device_type>
class BlobImpl : Blob {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BlobImpl);
  BlobImpl(Regst* regst, const BlobDesc* blob_desc, char* mem_ptr)
      : BlobImpl(regst, blob_desc, mem_ptr, nullptr) {}
  BlobImpl(Regst* regst, const BlobDesc* blob_desc, char* mem_ptr,
           const void* comm_net_token)
      : Blob(regst, blob_desc, mem_ptr, comm_net_token) {
    CHECK_EQ(NDIMS, blob_desc_ptr()->shape().NumAxes());
    for (int32_t d = 0; d < NDIMS; ++d) {
      dsizes_[d] = blob_desc_ptr()->shape().At(d);
    }
    tensor_ =
        EigenTensor<T, NDIMS>(reinterpret_cast<T*>(mut_memory_ptr()), dsizes_);
    const_tensor_ = EigenConstTensor<T, NDIMS>(
        reinterpret_cast<const T*>(memory_ptr()), dsizes_);
  }
  ~BlobImpl() = default;

  void Transpose(DeviceCtx* ctx, Blob* out_blob,
                 const std::vector<int32_t>& permutation) override {
    CHECK_EQ(NDIMS, out_blob->blob_desc_ptr()->shape().NumAxes());
    CHECK_EQ(blob_desc_ptr()->shape().elem_cnt(),
             out_blob->blob_desc_ptr()->shape().elem_cnt());
    Eigen::array<int32_t, NDIMS> p;
    for (int32_t i = 0; i < NDIMS; ++i) { p[i] = permutation[i]; }
    auto out_blob_impl =
        reinterpret_cast<BlobImpl<T, NDIMS, device_type>*>(out_blob);
    if (device_type == DeviceType::kCPU) {
      tensor_ = out_blob_impl->const_tensor_.shuffle(p);
    } else if (device_type == DeviceType::kGPU) {
      tensor_.device(ctx->eigen_gpu_device()) =
          out_blob_impl->const_tensor_.shuffle(p);
    } else {
      UNEXPECTED_RUN();
    }
  }

 private:
  EigenTensor<T, NDIMS> tensor_;
  EigenConstTensor<T, NDIMS> const_tensor_;
  Eigen::DSizes<Eigen::DenseIndex, NDIMS> dsizes_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_IMPLEMENT_H_
