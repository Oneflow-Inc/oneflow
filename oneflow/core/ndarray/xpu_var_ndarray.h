#ifndef ONEFLOW_CORE_NDARRAY_XPU_VAR_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_XPU_VAR_NDARRAY_H_

#include "oneflow/core/ndarray/exec_shape.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

template<typename T>
struct XpuVarNdarray final {
  explicit XpuVarNdarray(const Blob* blob)
      : shape_(blob->shape()), ptr_(blob->dptr<typename std::remove_const<T>::type>()) {}
  explicit XpuVarNdarray(Blob* blob) : shape_(blob->shape()), ptr_(blob->mut_dptr<T>()) {}
  OF_DEVICE_FUNC XpuVarNdarray(const XpuVarNdarray&) = default;
  OF_DEVICE_FUNC XpuVarNdarray(const ExecShape& shape, T* ptr) : shape_(shape), ptr_(ptr) {}

  OF_DEVICE_FUNC const ExecShape& shape() const { return shape_; }

  template<int NDIMS>
  OF_DEVICE_FUNC T Get(int64_t offset) const {
    return ptr_[offset];
  }

  template<int NDIMS>
  OF_DEVICE_FUNC T* Mut(int64_t offset) {
    return ptr_ + offset;
  }

 private:
  ExecShape shape_;
  T* ptr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_VAR_NDARRAY_H_
