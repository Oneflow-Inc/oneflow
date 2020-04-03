#ifndef ONEFLOW_CUSTOMIZED_KERNELS_WHERE_KERNEL_UTIL_H_
#define ONEFLOW_CUSTOMIZED_KERNELS_WHERE_KERNEL_UTIL_H_

#include "oneflow/core/device/device_context.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename CondT>
struct WhereKernelUtil {
  static void Where(DeviceCtx* ctx, const int64_t elem_cnt, const CondT* cond, const T* lhs,
                    const T* rhs, T* out);
};

#define INSTANTIATE_WHERE_FUNCTOR(device_type_v, dtype_pair, ctype_pair)       \
  template struct WhereKernelUtil<device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                  OF_PP_PAIR_FIRST(ctype_pair)>;

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_WHERE_KERNEL_UTIL_H_
