#ifndef ONEFLOW_CORE_REGISTER_BLOB_IMPLEMENT_H_
#define ONEFLOW_CORE_REGISTER_BLOB_IMPLEMENT_H_

#include "oneflow/core/eigen/tensor_type.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

template<DeviceType device_type, typename T, int32_t NDIMS>
struct BlobImplUtil {
  static void DoTranspose(DeviceCtx* ctx, EigenTensor<T, NDIMS>* tensor,
                          EigenConstTensor<T, NDIMS>* const_tensor,
                          const std::vector<int32_t>& permutation);
};

template<typename T, int32_t NDIMS, DeviceType device_type>
class BlobImpl : public Blob {
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
    tensor_ = of_make_unique<EigenTensor<T, NDIMS>>(
        reinterpret_cast<T*>(mut_memory_ptr()), dsizes_);
    const_tensor_ = of_make_unique<EigenConstTensor<T, NDIMS>>(
        reinterpret_cast<const T*>(memory_ptr()), dsizes_);
  }
  ~BlobImpl() = default;

  void Transpose(DeviceCtx* ctx, Blob* out_blob,
                 const std::vector<int32_t>& permutation) override {
    CHECK_EQ(NDIMS, out_blob->blob_desc_ptr()->shape().NumAxes());
    CHECK_EQ(blob_desc_ptr()->shape().elem_cnt(),
             out_blob->blob_desc_ptr()->shape().elem_cnt());
    auto out_blob_impl =
        reinterpret_cast<BlobImpl<T, NDIMS, device_type>*>(out_blob);
    BlobImplUtil<device_type, T, NDIMS>::DoTranspose(
        ctx, tensor_.get(), out_blob_impl->const_tensor_.get(), permutation);
  }

 private:
  std::unique_ptr<EigenTensor<T, NDIMS>> tensor_;
  std::unique_ptr<EigenConstTensor<T, NDIMS>> const_tensor_;
  Eigen::DSizes<Eigen::DenseIndex, NDIMS> dsizes_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_IMPLEMENT_H_
