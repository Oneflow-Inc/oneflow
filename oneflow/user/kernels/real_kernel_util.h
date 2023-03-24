#ifndef ONEFLOW_USER_KERNELS_REAL_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_REAL_KERNEL_UTIL_H_

namespace oneflow {
namespace user_op {

template<DeviceType device, typename dtype_x, typename dtype_out>
struct RealFunctor final {
  void operator()(ep::Stream* stream, const dtype_x* x, const dtype_out* out);
};

#define INSTANTIATE_REAL_FUNCTOR(device, dtype_x, dtype_out)   \
  template struct RealFunctor<device, dtype_x, dtype_out>;

} // namespace user_op
} // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_REAL_KERNEL_UTIL_H_
