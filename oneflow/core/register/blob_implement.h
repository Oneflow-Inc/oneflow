#ifndef ONEFLOW_CORE_REGISTER_BLOB_IMPLEMENT_H_
#define ONEFLOW_CORE_REGISTER_BLOB_IMPLEMENT_H_

#include "oneflow/core/eigen/tensor_type.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/eigen/eigen_tensor_interface.h"
#include "oneflow/core/eigen/eigen_tensor_implement.h"

namespace oneflow {

template<typename T, int32_t NDIMS, DeviceType device_type>
class BlobImpl<T, NDIMS> : Blob {
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
    tensor_ = Tensor<T, NDIMS>(reinterpret_cast<T*> mut_memory_ptr(), dsizes_);
    const_tensor_ =
        ConstTensor<T, NDIMS>(reinterpret_cast<const T*> mem_ptr(), dsizes_);
  }
  ~BlobImpl() = default;

  void Transpose(DeviceCtx* ctx, Blob* out_blob,
                 std::vector<int32_t> permutation) override {
    CHECK_EQ(NDIMS, out_blob->blob_desc_ptr()->shape().NumAxes());
    CHECK_EQ(blob_desc_ptr()->shape().elem_cnt(),
             out_blob->blob_desc_ptr()->shape().elem_cnt());
    Eigen::array<int32_t, NDIMS> p;
    for (int32_t i = 0; i < NDIMS; ++i) { p[i] = permutation[i]; }
    auto out_blob_impl = reinterpret_cast<BlobImpl<T, NDIMS>*> out_blob;
    *(GenEigenTensorIf<device_type>(ctx)) = const_tensor_.shuffle(p);
  }

 private:
  std::unique_ptr<EigenTensorIf> GenEigenTensorIf(DeviceCtx* ctx) {
    if (device_type == DeviceType::kCPU) {
      return new EigenTensorImpl<Tensor<T, NDIMS>>(&tensor_);
    } else if (device_type == DeviceType::kGPU) {
      return new EigenTensorImpl<
          TensorDevice<Tensor<T, NDIMS>, Eigen::GpuDevice>>(
          &(tensor_.device(ctx->eigen_gpu_device())));
    } else {
      UNEXPECTED_RUN();
    }
  }

  Tensor<T, NDIMS> tensor_;
  ConstTensor<T, NDIMS> const_tensor_;
  Eigen::DSizes<Eigen::DenseIndex, NDIMS> dsizes_;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_IMPLEMENT_H_
