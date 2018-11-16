#ifndef ONEFLOW_CORE_NDARRAY_XPU_NDARRAY_UTIL_H_
#define ONEFLOW_CORE_NDARRAY_XPU_NDARRAY_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/ndarray_reduce.h"
#include "oneflow/core/ndarray/xpu_reduced_ndarray.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
struct XpuNdArrayUtil final {
// SwitchReduce(SwitchCase(ndims), ...)
#define DEFINE_NDARRAY_REDUCE(func_name, NDIMS) NdArrayReduce<device_type, T, NDIMS>::func_name
  DEFINE_STATIC_SWITCH_FUNC(void, Reduce, DEFINE_NDARRAY_REDUCE, MAKE_NDIM_CTRV_SEQ(DIM_SEQ));
#undef DEFINE_NDARRAY_REDUCE
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_NDARRAY_UTIL_H_
