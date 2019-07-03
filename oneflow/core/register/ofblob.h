#ifndef ONEFLOW_CORE_REGISTER_OFBLOB_H_
#define ONEFLOW_CORE_REGISTER_OFBLOB_H_

#include "oneflow/core/register/blob.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

class OfBlob final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfBlob);
  OfBlob(DeviceCtx* device_ctx, Blob* blob) : device_ctx_(device_ctx), blob_(blob) {
    mem_case_.mutable_host_mem();
  }
  ~OfBlob() = default;

  int data_type() const { return blob_->data_type(); }
  size_t NumAxes() const { return blob_->shape().NumAxes(); }
  int64_t elem_cnt() const { return blob_->shape().elem_cnt(); }
  void CopyShapeTo(int64_t* ptr, int64_t num_axis) const;

  template<typename T>
  void AutoMemCopyTo(T* ptr, int64_t len) const;
  template<typename T>
  void AutoMemCopyFrom(const T* ptr, int64_t len) const;

 private:
  DeviceCtx* device_ctx_;
  Blob* blob_;
  MemoryCase mem_case_;
};

inline void OfBlob::CopyShapeTo(int64_t* ptr, int64_t num_axis) const {
  CHECK_EQ(num_axis, NumAxes());
  FOR_RANGE(int32_t, i, 0, num_axis) { ptr[i] = blob_->shape().At(i); }
}

template<typename T>
void OfBlob::AutoMemCopyTo(T* ptr, int64_t len) const {
  CHECK_EQ(blob_->shape().elem_cnt(), len);
  CHECK(blob_->data_type() == GetDataType<T>::value);
  AutoMemcpy(device_ctx_, ptr, blob_->dptr(), len * sizeof(T), mem_case_, blob_->mem_case());
}

template<typename T>
void OfBlob::AutoMemCopyFrom(const T* ptr, int64_t len) const {
  CHECK_EQ(blob_->shape().elem_cnt(), len);
  CHECK(blob_->data_type() == GetDataType<T>::value);
  AutoMemcpy(device_ctx_, blob_->mut_dptr(), ptr, len * sizeof(T), blob_->mem_case(), mem_case_);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_OFBLOB_H_
