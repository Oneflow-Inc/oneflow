#ifndef ONEFLOW_CORE_CPU_NDARRAY_NDARRAY_H_
#define ONEFLOW_CORE_CPU_NDARRAY_NDARRAY_H_

#include <climits>
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/ndarray/xpu_shape.h"

namespace oneflow {

template<typename T, int NDIMS>
class CpuNdarray {
 public:
  using dtype = T;
  static const int ndims = NDIMS;

  ALWAYS_INLINE const XpuShape& xpu_shape() const { return xpu_shape_; }

 protected:
  explicit CpuNdarray(const Shape& shape) : xpu_shape_(shape) {}
  explicit CpuNdarray(const XpuShape& xpu_shape) : xpu_shape_(xpu_shape) {}
  virtual ~CpuNdarray() = default;

 private:
  XpuShape xpu_shape_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CPU_NDARRAY_NDARRAY_H_
