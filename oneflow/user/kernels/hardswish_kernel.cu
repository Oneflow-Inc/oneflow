// #include "oneflow/user/kernels/hardswish_kernel.h"
// #include "oneflow/user/kernels/elementwise_xpu_kernel.cuh"

// namespace oneflow {

// template<>
// struct HardswishFunctor<half> {
//   HardswishFunctor<float> float_functor;
//   OF_DEVICE_FUNC half operator()(half x) const {
//     return __float2half(float_functor(__half2float(x)));
//   }
// };
// template<>
// struct HardswishGradFunctor<half> {
//   HardswishGradFunctor<float> float_functor;
//   OF_DEVICE_FUNC half operator()(half x, half dy) const {
//     return __float2half(float_functor(__half2float(x), __half2float(dy)));
//   }
// };

// #define INSTANTIATE_HARDSWISH_GPU_FUNCTORS(dtype)                           \
//   INSTANTIATE_UNARY_XPU_FUNCTOR(DeviceType::kGPU, HardswishFunctor, dtype); \
//   INSTANTIATE_BINARY_XPU_FUNCTOR(DeviceType::kGPU, HardswishGradFunctor, dtype);

// INSTANTIATE_HARDSWISH_GPU_FUNCTORS(half);
// INSTANTIATE_HARDSWISH_GPU_FUNCTORS(double);
// INSTANTIATE_HARDSWISH_GPU_FUNCTORS(float);

// REGISTER_HARDSWISH_KERNEL(DeviceType::kGPU, half);
// REGISTER_HARDSWISH_KERNEL(DeviceType::kGPU, float);
// REGISTER_HARDSWISH_KERNEL(DeviceType::kGPU, double);

// } //namespace oneflow 