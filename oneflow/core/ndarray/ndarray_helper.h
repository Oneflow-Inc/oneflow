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
  template<int CONCAT_AXES, typename XT, typename T = typename XT::dtype, int NDIMS = XT::ndims>
  ConcatVarNdArray<T, NDIMS, CONCAT_AXES> Concatenate(const std::vector<XT>& var_ndarrays) const {
    static_assert(std::is_same<XT, VarNdArray<T, NDIMS>>::value,
                  "only vector of VarNdArray can be Concatenated");
    return ConcatVarNdArray<T, NDIMS, CONCAT_AXES>(var_ndarrays);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_HELPER_H_
