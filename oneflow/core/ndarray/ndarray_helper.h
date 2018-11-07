#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_HELPER_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_HELPER_H_

#include "oneflow/core/ndarray/var_ndarray.h"
#include "oneflow/core/ndarray/slice_ndarray.h"

namespace oneflow {

template<typename default_data_type, int default_ndims>
class NdArrayHelper final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NdArrayHelper);
  NdArrayHelper() = default;
  ~NdArrayHelper() = default;

  template<typename T = default_data_type, int NDIMS = default_ndims>
  VarNdArray<T, NDIMS> Var(const Shape& shape, T* ptr) {
    return VarNdArray<T, NDIMS>(shape, ptr);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_HELPER_H_
