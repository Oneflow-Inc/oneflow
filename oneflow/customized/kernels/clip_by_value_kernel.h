#ifndef ONEFLOW_CUSTOMIZED_KERNELS_CLIP_BY_VALUE_KERNEL_H_
#define ONEFLOW_CUSTOMIZED_KERNELS_CLIP_BY_VALUE_KERNEL_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/xpu_util.h"

namespace oneflow {

template<typename T>
struct ClipByMinFunctor {
  template<typename U>
  ClipByMinFunctor(U min) : min_value_(static_cast<T>(min)) {}
  T operator()(T value) {
#if defined(__CUDA_ARCH__)
    return max(value, min_value_);
#else
    return std::max(value, min_value_);
#endif
  }
  T min_value_;
};

template<typename T>
struct ClipByMaxFunctor {
  template<typename U>
  ClipByMaxFunctor(U max) : max_value_(static_cast<T>(max)) {}
  T operator()(T value) {
#if defined(__CUDA_ARCH__)
    return min(value, max_value_);
#else
    return std::min(value, max_value_);
#endif
  }
  T max_value_;
};

template<typename T>
struct ClipByMinMaxFunctor {
  template<typename U>
  ClipByMinMaxFunctor(U min, U max)
      : min_value_(static_cast<T>(min)), max_value_(static_cast<T>(max)) {}
  T operator()(T value) {
#if defined(__CUDA_ARCH__)
    return min(max(value, min_value_), max_value_);
#else
    return std::min(std::max(value, min_value_), max_value_);
#endif
  }
  T min_value_;
  T max_value_;
};

template<typename T>
struct ClipByMinGradFunctor {
  template<typename U>
  ClipByMinGradFunctor(U min) : min_value_(static_cast<T>(min)) {}
  T operator()(T value, T grad) { return value < min_value_ ? static_cast<T>(0) : grad; }
  T min_value_;
};

template<typename T>
struct ClipByMaxGradFunctor {
  template<typename U>
  ClipByMaxGradFunctor(U max) : max_value_(static_cast<T>(max)) {}
  T operator()(T value, T grad) { return value > max_value_ ? static_cast<T>(0) : grad; }
  T max_value_;
};

template<typename T>
struct ClipByMinMaxGradFunctor {
  template<typename U>
  ClipByMinMaxGradFunctor(U min, U max)
      : min_value_(static_cast<T>(min)), max_value_(static_cast<T>(max)) {}
  T operator()(T value, T grad) {
    return (value < min_value_ || value > max_value_) ? static_cast<T>(0) : grad;
  }
  T min_value_;
  T max_value_;
};

template<typename T>
struct ClipUtil {
  template<typename F>
  OF_DEVICE_FUNC static void Forward(F f, const int64_t num_values, const T* x, T* y) {
    XPU_1D_KERNEL_LOOP(i, num_values) { y[i] = f(x[i]); }
  }

  template<typename F>
  OF_DEVICE_FUNC static void Backward(F f, const int64_t num_values, const T* x, const T* dy,
                                      T* dx) {
    XPU_1D_KERNEL_LOOP(i, num_values) { dx[i] = f(x[i], dy[i]); }
  }
};

template<DeviceType device_type, typename T>
class ClipByScalarKernel;

template<DeviceType device_type, typename T>
class ClipByScalarMinKernel;

template<DeviceType device_type, typename T>
class ClipByScalarMaxKernel;

template<DeviceType device_type, typename T>
class ClipByScalarGradKernel;

template<DeviceType device_type, typename T>
class ClipByScalarMinGradKernel;

template<DeviceType device_type, typename T>
class ClipByScalarMaxGradKernel;

#define REGISTER_CLIP_KERNEL(op_type_name, kernel_name, device_type_v, dtype)                   \
  REGISTER_USER_KERNEL(#op_type_name)                                                           \
      .SetCreateFn([](user_op::KernelInitContext* ctx) {                                        \
        return new kernel_name##Kernel<device_type_v, dtype>(ctx);                              \
      })                                                                                        \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) -> bool {                      \
        return (ctx.device_type() == device_type_v                                              \
                && ctx.TensorDesc4ArgNameAndIndex("y", 0)->data_type()                          \
                       == GetDataType<dtype>::value);                                           \
      })                                                                                        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "x", 0, true));                          \
        return Maybe<void>::Ok();                                                               \
      });

#define REGISTER_CLIP_GRAD_KERNEL(op_type_name, kernel_name, device_type_v, dtype)              \
  REGISTER_USER_KERNEL(#op_type_name)                                                           \
      .SetCreateFn([](user_op::KernelInitContext* ctx) {                                        \
        return new kernel_name##GradKernel<device_type_v, dtype>(ctx);                          \
      })                                                                                        \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) -> bool {                      \
        return (ctx.device_type() == device_type_v                                              \
                && ctx.TensorDesc4ArgNameAndIndex("dx", 0)->data_type()                         \
                       == GetDataType<dtype>::value);                                           \
      })                                                                                        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dy", 0, "dx", 0, true));                        \
        return Maybe<void>::Ok();                                                               \
      });

#define REGISTER_CLIP_KERNELS(device_type_v, dtype_pair)                                          \
  REGISTER_CLIP_KERNEL(clip_by_scalar, ClipByScalar, device_type_v, OF_PP_PAIR_FIRST(dtype_pair)) \
  REGISTER_CLIP_KERNEL(clip_by_scalar_min, ClipByScalarMin, device_type_v,                        \
                       OF_PP_PAIR_FIRST(dtype_pair))                                              \
  REGISTER_CLIP_KERNEL(clip_by_scalar_max, ClipByScalarMax, device_type_v,                        \
                       OF_PP_PAIR_FIRST(dtype_pair))                                              \
  REGISTER_CLIP_GRAD_KERNEL(clip_by_scalar_grad, ClipByScalar, device_type_v,                     \
                            OF_PP_PAIR_FIRST(dtype_pair))                                         \
  REGISTER_CLIP_GRAD_KERNEL(clip_by_scalar_min_grad, ClipByScalarMin, device_type_v,              \
                            OF_PP_PAIR_FIRST(dtype_pair))                                         \
  REGISTER_CLIP_GRAD_KERNEL(clip_by_scalar_max_grad, ClipByScalarMax, device_type_v,              \
                            OF_PP_PAIR_FIRST(dtype_pair))

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_CLIP_BY_VALUE_KERNEL_H_
