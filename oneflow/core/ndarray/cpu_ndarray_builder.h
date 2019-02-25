#ifndef ONEFLOW_CORE_CPU_NDARRAY_NDARRAY_HELPER_H_
#define ONEFLOW_CORE_CPU_NDARRAY_NDARRAY_HELPER_H_

#include "oneflow/core/ndarray/cpu_var_ndarray.h"
#include "oneflow/core/ndarray/cpu_slice_var_ndarray.h"
#include "oneflow/core/ndarray/cpu_concat_var_ndarray.h"

namespace oneflow {

template<typename default_data_type, int default_ndims>
class CpuNdArrayBuilder final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuNdArrayBuilder);
  CpuNdArrayBuilder() = default;
  ~CpuNdArrayBuilder() = default;

  template<typename T = default_data_type, int NDIMS = default_ndims>
  CpuVarNdArray<T, NDIMS> Var(const Shape& shape, T* ptr) const {
    return CpuVarNdArray<T, NDIMS>(shape, ptr);
  }
  template<int CONCAT_AXES = 0, typename T = default_data_type, int NDIMS = default_ndims>
  CpuConcatVarNdArray<T, NDIMS, CONCAT_AXES> Concatenate(
      const std::vector<CpuVarNdArray<T, NDIMS>>& var_ndarrays) const {
    return CpuConcatVarNdArray<T, NDIMS, CONCAT_AXES>(var_ndarrays);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CPU_NDARRAY_NDARRAY_HELPER_H_
