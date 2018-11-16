#ifndef ONEFLOW_CORE_NDARRAY_XPU_REDUCED_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_XPU_REDUCED_NDARRAY_H_

#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/unary_func.h"

namespace oneflow {

template<typename T, int NDIMS>
class XpuReducedNdarray final {
 public:
  OF_DEVICE_FUNC XpuReducedNdarray(const ExecShape& shape, const XpuVarNdarray<T>& data)
      : shape_(shape), data_(data) {}

  OF_DEVICE_FUNC const ExecShape& shape() const { return shape_; }
  const ExecShape& host_shape() const { return shape_; }
  OF_DEVICE_FUNC const XpuVarNdarray<T>& data() const { return data_; }

  template<int ndims = NDIMS>
  OF_DEVICE_FUNC T Get(int64_t offset) const {
    int64_t coord[NDIMS];
    Offset2Coord(offset, coord);
    return Get(coord);
  }

  template<int ndims = NDIMS>
  OF_DEVICE_FUNC T Get(int64_t coord[NDIMS]) const {
    return data_.template Get<NDIMS>(coord);
  }

  template<int ndims = NDIMS>
  OF_DEVICE_FUNC T* Mut(int64_t offset) const {
    int64_t coord[NDIMS];
    Offset2Coord(offset, coord);
    return Mut(coord);
  }

  template<int ndims = NDIMS>
  OF_DEVICE_FUNC T* Mut(int64_t coord[NDIMS]) const {
    return data_.template Mut<NDIMS>(coord);
  }

  OF_DEVICE_FUNC void Offset2Coord(int64_t offset, int64_t coord[NDIMS]) const {
    ExecShapeUtil<NDIMS>::Offset2DimVec(shape_, offset, coord);
  }

 private:
  ExecShape shape_;
  XpuVarNdarray<T> data_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_REDUCED_NDARRAY_H_
