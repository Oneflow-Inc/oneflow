#ifndef ONEFLOW_CORE_REGISTER_BLOB_IMPLEMENT_H_
#define ONEFLOW_CORE_REGISTER_BLOB_IMPLEMENT_H_

#include "oneflow/core/register/blob.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type>
class BlobImpl final : public Blob {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BlobImpl);
  BlobImpl(Regst* regst, const BlobDesc* blob_desc, char* mem_ptr)
      : Blob(regst, blob_desc, mem_ptr) {}
  ~BlobImpl() = default;

  void CopyDataContentFrom(DeviceCtx* device_ctx, const Blob* rhs) override {
    if (this == rhs) { return; }
    Memcpy<device_type>(device_ctx, mut_dptr(), rhs->dptr(), ByteSizeOfDataContentField());
  }
  void CopyDataIdFrom(DeviceCtx* device_ctx, const Blob* rhs) override {
    if (this == rhs) { return; }
    Memcpy<device_type>(device_ctx, mut_data_id(), rhs->data_id(), ByteSizeOfDataIdField());
  }
  void CopyColNumFrom(DeviceCtx* device_ctx, const Blob* rhs) override {
    if (this == rhs) { return; }
    Memcpy<device_type>(device_ctx, mut_col_num(), rhs->col_num(), ByteSizeOfColNumField());
  }
  void CopyFrom(DeviceCtx* device_ctx, const Blob* rhs) override {
    if (this == rhs) { return; }
    Memcpy<device_type>(device_ctx, mut_memory_ptr(), rhs->memory_ptr(), TotalByteSize());
  }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_BLOB_IMPLEMENT_H_
