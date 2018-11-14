#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_HELPER_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_HELPER_H_

#include "oneflow/core/ndarray/var_ndarray.h"
#include "oneflow/core/ndarray/slice_ndarray.h"
#include "oneflow/core/ndarray/concat_var_ndarray.h"

namespace oneflow {

template<typename default_data_type, int default_ndims>
class NdArrayHelper final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NdArrayHelper);
  NdArrayHelper() = default;
  ~NdArrayHelper() = default;

  template<typename T = default_data_type, int NDIMS = default_ndims>
  VarNdArray<T, NDIMS> Var(const Shape& shape, T* ptr) const {
    return VarNdArray<T, NDIMS>(shape, ptr);
  }
  template<int CONCAT_AXES = 0, typename T = default_data_type, int NDIMS = default_ndims>
  ConcatVarNdArray<T, NDIMS, CONCAT_AXES> Concatenate(
      const std::vector<VarNdArray<T, NDIMS>>& var_ndarrays) const {
    return ConcatVarNdArray<T, NDIMS, CONCAT_AXES>(var_ndarrays);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_HELPER_H_
