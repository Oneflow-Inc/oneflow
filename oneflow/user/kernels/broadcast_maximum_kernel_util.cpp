#include "oneflow/user/kernels/broadcast_maximum_kernel_util.h"

namespace oneflow{
namespace user_op{


template<typename T>
struct MaximumBackwardFunctor<DeviceType::kCPU, T> final {
  void operator()(DeviceCtx* ctx, int64_t elem_cnt, 
                const T* dz, const T* x,  const T* y, 
                T* dx, T* dy){
      DoUpdateMaximumGrad(elem_cnt, dz, x, y, dx, dy);
  }
};

template struct MaximumBackwardFunctor<DeviceType::kCPU, float>;
template struct MaximumBackwardFunctor<DeviceType::kCPU, double>;

} //namespace user_op
}// namespace oneflow