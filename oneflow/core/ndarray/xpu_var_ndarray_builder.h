#ifndef ONEFLOW_CORE_NDARRAY_XPU_VAR_NDARRAY_BUILDER_H_
#define ONEFLOW_CORE_NDARRAY_XPU_VAR_NDARRAY_BUILDER_H_

#include "oneflow/core/ndarray/xpu_var_ndarray.h"

namespace oneflow {

template<typename T>
class XpuVarNdarrayBuilder final {
 public:
  XpuVarNdarrayBuilder() = default;
  XpuVarNdarrayBuilder(const XpuVarNdarrayBuilder&) = default;
  ~XpuVarNdarrayBuilder() = default;

  XpuVarNdarray<T> operator()(const Shape& shape, T* ptr) const {
    return XpuVarNdarray<T>(shape, ptr);
  }
  template<typename DT = T>
  typename std::enable_if<!std::is_same<DT, const DT>::value, XpuVarNdarray<DT>>::type 
  operator()(Blob* blob, int ndims_extend_to) const {
    return XpuVarNdarray<DT>(blob, ndims_extend_to);
  }
  template<typename DT = T>
  typename std::enable_if<!std::is_same<DT, const DT>::value, XpuVarNdarray<const DT>>::type 
  operator()(const Blob* blob, int ndims_extend_to) const {
    return XpuVarNdarray<const DT>(blob, ndims_extend_to);
  }
};

}

#endif  // ONEFLOW_CORE_NDARRAY_XPU_VAR_NDARRAY_BUILDER_H_
