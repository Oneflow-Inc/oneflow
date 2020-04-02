#ifndef ONEFLOW_CUSTOMIZED_KERNELS_WHERE_KERNEL_UTIL_H_
#define ONEFLOW_CUSTOMIZED_KERNELS_WHERE_KERNEL_UTIL_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/ndarray/xpu_util.h"

namespace oneflow {

template<typename T, typename CondT>
OF_DEVICE_FUNC void DoWhere(const int64_t elem_cnt, const CondT* cond, const T* lhs, const T* rhs,
                            T* out) {
  XPU_1D_KERNEL_LOOP(i, elem_cnt) { out[i] = static_cast<bool>(cond[i]) ? lhs[i] : rhs[i]; }
}

template<DeviceType device_type, typename T, typename CondT>
struct WhereFunctor {
  void operator()(DeviceCtx* ctx, const int64_t elem_cnt, const CondT* cond, const T* lhs,
                  const T* rhs, T* out) const;
};

#define INSTANTIATE_WHERE_FUNCTOR(device_type_v, dtype_pair, ctype_pair)    \
  template struct WhereFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                               OF_PP_PAIR_FIRST(ctype_pair)>;

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_WHERE_KERNEL_UTIL_H_
