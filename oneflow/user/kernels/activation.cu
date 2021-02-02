#include "oneflow/user/kernels/activation.h"
#include "oneflow/user/kernels/elementwise_xpu_kernel.cuh"

namespace oneflow {

template<>
struct EluFunctor<half> {
  OF_DEVICE_FUNC explicit EluFunctor(float alpha) : alpha(alpha), float_functor(EluFunctor<float>(alpha)) {}
  OF_DEVICE_FUNC half operator()(half x) const { return __float2half(float_functor(__half2float(x))); }
  const float alpha;
  EluFunctor<float> float_functor;
};

template<>
struct EluGradFunctor<half> {
  OF_DEVICE_FUNC explicit EluGradFunctor(float alpha)
      : alpha(alpha), float_functor(EluGradFunctor<float>(alpha)) {}
  OF_DEVICE_FUNC half operator()(half x, half dy) const {
    return __float2half(float_functor(__half2float(x), __half2float(dy)));
  }
  const float alpha;
  EluGradFunctor<float> float_functor;
};

template<>
struct HardswishFunctor<half> {
  HardswishFunctor<float> float_functor;
  OF_DEVICE_FUNC half operator()(half x) const {
    return __float2half(float_functor(__half2float(x)));
  }
};

template<>
struct HardswishGradFunctor<half> {
  HardswishGradFunctor<float> float_functor;
  OF_DEVICE_FUNC half operator()(half x, half dy) const {
    return __float2half(float_functor(__half2float(x), __half2float(dy)));
  }
};

#define INSTANTIATE_GPU_ACTIVATION_FORWARD_FUNCTORS(dtype) \
  INSTANTIATE_UNARY_XPU_FUNCTOR(DeviceType::kGPU, EluFunctor, dtype); \
  INSTANTIATE_UNARY_XPU_FUNCTOR(DeviceType::kGPU, HardswishFunctor, dtype);  

#define INSTANTIATE_GPU_ACTIVATION_BACKWARD_FUNCTORS(dtype) \
  INSTANTIATE_BINARY_XPU_FUNCTOR(DeviceType::kGPU, EluGradFunctor, dtype); \
  INSTANTIATE_BINARY_XPU_FUNCTOR(DeviceType::kGPU, HardswishGradFunctor, dtype);


#define INSTANTIATE_GPU_ACTIVATION_FUNCTORS(dtype)                           \
  INSTANTIATE_GPU_ACTIVATION_FORWARD_FUNCTORS(dtype)                           \
  INSTANTIATE_GPU_ACTIVATION_BACKWARD_FUNCTORS(dtype)                           
  

INSTANTIATE_GPU_ACTIVATION_FUNCTORS(half);
INSTANTIATE_GPU_ACTIVATION_FUNCTORS(double);
INSTANTIATE_GPU_ACTIVATION_FUNCTORS(float);


#define REGISTER_ACTIVATION_GPU_KERNEL(dtype)   \
  REGISTER_ELU_KERNEL(DeviceType::kGPU, dtype); \
  REGISTER_HARDSWISH_KERNEL(DeviceType::kGPU, dtype);

REGISTER_ACTIVATION_GPU_KERNEL(float);
REGISTER_ACTIVATION_GPU_KERNEL(double);

} //namespace oneflow 