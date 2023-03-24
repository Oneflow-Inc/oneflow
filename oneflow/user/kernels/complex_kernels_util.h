#ifndef ONEFLOW_USER_KERNELS_COMPLEX_KERNELS_UTIL_H_
#define ONEFLOW_USER_KERNELS_COMPLEX_KERNELS_UTIL_H_

namespace oneflow {
namespace user_op {

template<DeviceType device, typename dtype_x, typename dtype_out>
struct RealFunctor final {
  void operator()(ep::Stream* stream, const dtype_x* x, const dtype_out* out);
};

#define INSTANTIATE_REAL_FUNCTOR(device, dtype_x, dtype_out)   \
  template struct RealFunctor<device, dtype_x, dtype_out>;

template<DeviceType device, typename dtype_x, typename dtype_out>
struct ImagFunctor final {
  void operator()(ep::Stream* stream, const dtype_x* x, const dtype_out* out);
};

#define INSTANTIATE_IMAG_FUNCTOR(device, dtype_x, dtype_out)   \
  template struct ImagFunctor<device, dtype_x, dtype_out>;

template<DeviceType device, typename dtype>
struct ConjPhysicalFunctor final {
  void operator()(ep::Stream* stream, const dtype* x, const dtype* out);
};

#define INSTANTIATE_CONJ_PHYSICAL_FUNCTOR(device, dtype)   \
  template struct ConjPhysicalFunctor<device, dtype>;

} // namespace user_op
} // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_COMPLEX_KERNELS_UTIL_H_
