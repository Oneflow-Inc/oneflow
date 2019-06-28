#ifndef ONEFLOW_CORE_REGISTER_FOREIGN_BLOB_H_
#define ONEFLOW_CORE_REGISTER_FOREIGN_BLOB_H_

#include "oneflow/core/register/blob.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

class ForeignBlob final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ForeignBlob);
  ForeignBlob(DeviceCtx* device_ctx, Blob* blob) : device_ctx_(device_ctx), blob_(blob) {
    mem_case_.mutable_host_mem();
  }
  ~ForeignBlob() = default;

  int data_type() const { return blob_->data_type(); }
  size_t NumAxes() const { return blob_->shape().NumAxes(); }
  void CopyShapeTo(int64_t* ptr, int64_t num_axis) const;

#define DEFINE_COPIER(T, type_proto)                                           \
  void CopyToBuffer(T* ptr, int64_t len) const { AutoMemCopyTo<T>(ptr, len); } \
  void CopyFromBuffer(const T* ptr, int64_t len) const { AutoMemCopyFrom<T>(ptr, len); }

  OF_PP_FOR_EACH_TUPLE(DEFINE_COPIER, POD_DATA_TYPE_SEQ);

#undef DEFINE_COPIER

 private:
  template<typename T>
  void AutoMemCopyTo(T* ptr, int64_t len) const;
  template<typename T>
  void AutoMemCopyFrom(const T* ptr, int64_t len) const;

  DeviceCtx* device_ctx_;
  Blob* blob_;
  MemoryCase mem_case_;
};

inline void ForeignBlob::CopyShapeTo(int64_t* ptr, int64_t num_axis) const {
  CHECK_EQ(num_axis, NumAxes());
  FOR_RANGE(int32_t, i, 0, num_axis) { ptr[i] = blob_->shape().At(i); }
}

template<typename T>
void ForeignBlob::AutoMemCopyTo(T* ptr, int64_t len) const {
  CHECK_EQ(blob_->shape().elem_cnt(), len);
  CHECK(blob_->data_type() == GetDataType<T>::value);
  AutoMemcpy(device_ctx_, ptr, blob_->dptr(), len * sizeof(T), mem_case_, blob_->mem_case());
}

template<typename T>
void ForeignBlob::AutoMemCopyFrom(const T* ptr, int64_t len) const {
  CHECK_EQ(blob_->shape().elem_cnt(), len);
  CHECK(blob_->data_type() == GetDataType<T>::value);
  AutoMemcpy(device_ctx_, blob_->mut_dptr(), ptr, len * sizeof(T), blob_->mem_case(), mem_case_);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_FOREIGN_BLOB_H_
