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
  explicit BlobImpl(Regst* regst, const BlobDesc* blob_desc, char* header_mem_ptr,
                    char* data_mem_ptr)
      : Blob(regst, blob_desc, header_mem_ptr, data_mem_ptr) {
    CHECK_EQ(NDIMS, blob_desc_ptr()->shape().NumAxes());
    for (int32_t d = 0; d < NDIMS; ++d) { dsizes_[d] = blob_desc_ptr()->shape().At(d); }
    tensor_ = std::make_unique<EigenTensor<T, NDIMS>>(mut_dptr<T>(), dsizes_);
    const_tensor_ = std::make_unique<EigenConstTensor<T, NDIMS>>(dptr<T>(), dsizes_);
  }
  ~BlobImpl() = default;

  void CopyDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs) override {
    if (this == rhs) { return; }
    AutoMemcpy(device_ctx, mut_dptr(), rhs->dptr(), ByteSizeOfDataContentField(), mem_case(),
               rhs->mem_case());
  }
  void CopyDataIdFrom(DeviceCtx* device_ctx, const Blob* rhs) override {
    if (this == rhs) { return; }
    Memcpy<DeviceType::kCPU>(device_ctx, mut_data_id(), rhs->data_id(), ByteSizeOfDataIdField());
  }
  void CopyColNumFrom(DeviceCtx* device_ctx, const Blob* rhs) override {
    if (this == rhs) { return; }
    Memcpy<DeviceType::kCPU>(device_ctx, mut_col_num(), rhs->col_num(), ByteSizeOfColNumField());
  }
  void CopyHeaderFrom(DeviceCtx* device_ctx, const Blob* rhs) override {
    if (this == rhs) { return; }
    size_t header_size = ByteSizeOfHeaderField();
    CHECK(header_size > 0);
    Memcpy<DeviceType::kCPU>(device_ctx, mut_hptr(), rhs->hptr(), header_size);
  }
  void CopyFrom(DeviceCtx* device_ctx, const Blob* rhs) override {
    if (this == rhs) { return; }
    if (device_type == DeviceType::kCPU) {
      Memcpy<DeviceType::kCPU>(device_ctx, mut_hptr(), rhs->hptr(), TotalByteSize());
    } else {
      if (ByteSizeOfHeaderField() > 0) { CopyHeaderFrom(device_ctx, rhs); }
      CopyDataContentFrom(device_ctx, rhs);
    }
  }

 private:
  std::unique_ptr<EigenTensor<T, NDIMS>> tensor_;
  std::unique_ptr<EigenConstTensor<T, NDIMS>> const_tensor_;
  Eigen::DSizes<Eigen::DenseIndex, NDIMS> dsizes_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_IMPLEMENT_H_
