#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_H_

#include <climits>
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/ndarray/xpu_shape.h"

namespace oneflow {

template<typename T, int NDIMS>
class CpuNdArray {
 public:
  using dtype = T;
  static const int ndims = NDIMS;

  ALWAYS_INLINE const XpuShape& xpu_shape() const { return xpu_shape_; }

 protected:
  explicit CpuNdArray(const Shape& shape) : xpu_shape_(shape) {}
  explicit CpuNdArray(const XpuShape& xpu_shape) : xpu_shape_(xpu_shape) {}
  virtual ~CpuNdArray() = default;

 private:
  XpuShape xpu_shape_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_H_
