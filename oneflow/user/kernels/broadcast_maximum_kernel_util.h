#ifndef ONEFLOW_USER_KERNELS_BROADCAST_MAXIMUM_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_BROADCAST_MAXIMUM_KERNEL_UTIL_H_
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow{
namespace user_op{

template<DeviceType device_type, typename T>
struct MaximumBackwardFunctor final {
  void operator()(DeviceCtx* ctx, int64_t elem_cnt, 
                const T* dz, const T* x,  const T* y, 
                T* dx, T* dy);
};

template<typename T>
OF_DEVICE_FUNC void DoUpdateMaximumGrad(int64_t elem_cnt, 
                                     const T* dz,
                                     const T* x,  const T* y,
                                     T* dx, T* dy) {
  XPU_1D_KERNEL_LOOP(idx, elem_cnt) {
    if(x[idx] > y[idx]){
      dx[idx] = dz[idx];
    }else{
      dy[idx] = dz[idx];
    }
  }
}

} //namespace user_op
}// namespace oneflow

#endif // ONEFLOW_USER_KERNELS_BROADCAST_MAXIMUM_KERNEL_H_