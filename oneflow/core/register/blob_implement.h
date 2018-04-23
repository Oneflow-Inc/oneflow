#ifndef ONEFLOW_CORE_REGISTER_BLOB_IMPLEMENT_H_
#define ONEFLOW_CORE_REGISTER_BLOB_IMPLEMENT_H_

#include "oneflow/core/common/eigen_util.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, int32_t NDIMS>
struct BlobImplUtil {
  static void DoTranspose(DeviceCtx* ctx, EigenTensor<T, NDIMS>* tensor,
                          EigenConstTensor<T, NDIMS>* const_tensor,
                          const PbRf<int32_t>& permutation);
};

template<typename T, int32_t NDIMS, DeviceType device_type>
class BlobImpl final : public Blob {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BlobImpl);
  BlobImpl(Regst* regst, const BlobDesc* blob_desc, char* head_mem_ptr,
           char* body_mem_ptr, const void* comm_net_token)
      : Blob(regst, blob_desc, head_mem_ptr, body_mem_ptr, comm_net_token) {
    CHECK_EQ(NDIMS, blob_desc_ptr()->shape().NumAxes());
    for (int32_t d = 0; d < NDIMS; ++d) {
      dsizes_[d] = blob_desc_ptr()->shape().At(d);
    }
    tensor_ = of_make_unique<EigenTensor<T, NDIMS>>(
        reinterpret_cast<T*>(mut_dptr()), dsizes_);
    const_tensor_ = of_make_unique<EigenConstTensor<T, NDIMS>>(
        reinterpret_cast<const T*>(dptr()), dsizes_);
  }
  ~BlobImpl() = default;

  void CopyDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs) override {
    if (this == rhs) { return; }
    Memcpy<device_type>(device_ctx, mut_dptr(), rhs->dptr(),
                        ByteSizeOfDataContentField());
  }
  void CopyDataIdFrom(DeviceCtx* device_ctx, const Blob* rhs) override {
    if (this == rhs) { return; }
    Memcpy<DeviceType::kCPU>(device_ctx, mut_data_id(), rhs->data_id(),
                             ByteSizeOfDataIdField());
  }
  void CopyColNumFrom(DeviceCtx* device_ctx, const Blob* rhs) override {
    if (this == rhs) { return; }
    Memcpy<DeviceType::kCPU>(device_ctx, mut_col_num(), rhs->col_num(),
                             ByteSizeOfColNumField());
  }
  void CopyFrom(DeviceCtx* device_ctx, const Blob* rhs) override {
    if (this == rhs) { return; }
    if (IsContinues()) {
      Memcpy<device_type>(device_ctx, mut_head_memory_ptr(),
                          rhs->head_memory_ptr(), TotalByteSize());
    } else {
      CopyDataContentFrom(device_ctx, rhs);
      CopyDataIdFrom(device_ctx, rhs);
      CopyColNumFrom(device_ctx, rhs);
    }
  }

 private:
  std::unique_ptr<EigenTensor<T, NDIMS>> tensor_;
  std::unique_ptr<EigenConstTensor<T, NDIMS>> const_tensor_;
  Eigen::DSizes<Eigen::DenseIndex, NDIMS> dsizes_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_IMPLEMENT_H_
