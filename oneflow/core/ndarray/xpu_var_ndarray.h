#ifndef ONEFLOW_CORE_NDARRAY_XPU_VAR_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_XPU_VAR_NDARRAY_H_

#include "oneflow/core/ndarray/exec_shape.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<typename T>
class XpuVarNdarray final {
 public:
  explicit XpuVarNdarray(const Blob* blob, int ndims_extend_to)
      : shape_(blob->shape().CreateLeftExtendedShape(ndims_extend_to)),
        ptr_(blob->dptr<typename std::remove_const<T>::type>()) {}
  explicit XpuVarNdarray(Blob* blob, int ndims_extend_to)
      : shape_(blob->shape().CreateLeftExtendedShape(ndims_extend_to)), ptr_(blob->mut_dptr<T>()) {}
  XpuVarNdarray(const Shape& shape, T* ptr) : shape_(shape), ptr_(ptr) {}
  OF_DEVICE_FUNC ALWAYS_INLINE XpuVarNdarray(const XpuVarNdarray&) = default;
  OF_DEVICE_FUNC ALWAYS_INLINE XpuVarNdarray(const ExecShape& shape, T* ptr)
      : shape_(shape), ptr_(ptr) {}

  const ExecShape& host_shape() const { return shape_; }
  T* host_ptr() const { return ptr_; }

  OF_DEVICE_FUNC const ExecShape& shape() const { return shape_; }
  OF_DEVICE_FUNC T* ptr() const { return ptr_; }

  template<int NDIMS>
  OF_DEVICE_FUNC T Get(int64_t offset) const {
    return ptr_[offset];
  }
  template<int NDIMS>
  OF_DEVICE_FUNC T Get(int64_t coord[NDIMS]) const {
    return ptr_[ExecShapeUtil<NDIMS>::DimVec2Offset(shape(), coord)];
  }

  template<int NDIMS>
  OF_DEVICE_FUNC T* Mut(int64_t offset) const {
    return ptr_ + offset;
  }

  template<int NDIMS>
  OF_DEVICE_FUNC T* Mut(int64_t coord[NDIMS]) const {
    return ptr_ + ExecShapeUtil<NDIMS>::DimVec2Offset(shape(), coord);
  }

  template<int NDIMS, typename X>
  OF_DEVICE_FUNC void Assign(const X& x) const {
    AssignWithoutSyncThreads<NDIMS>(x);
    XpuSyncThreads();
  }

  template<int NDIMS, typename X>
  OF_DEVICE_FUNC void AssignWithoutSyncThreads(const X& x) const {
    size_t n = shape_.ElemNum();
    XPU_1D_KERNEL_LOOP(i, n) { ptr_[i] = x.template Get<NDIMS>(i); }
  }

 private:
  ExecShape shape_;
  T* ptr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_VAR_NDARRAY_H_
