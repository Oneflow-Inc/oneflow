#ifndef _ONEFLOW_USER_KERNELS_ELEMENTWISE_ELU_KERNEL_H_
#define _ONEFLOW_USER_KERNELS_ELEMENTWISE_ELU_KERNEL_H_
namespace oneflow {

template<typename T>
struct EluFunctor {
  OF_DEVICE_FUNC explicit EluFunctor(T alpha) : alpha(alpha) {}
  OF_DEVICE_FUNC T operator()(T x) const {
    return (x > static_cast<T>(0)) ? x : alpha * (exp(x) - static_cast<T>(1));
  }
  const T alpha;
};

} //namespace oneflow 

#endif  // _ONEFLOW_USER_KERNELS_ELEMENTWISE_ELU_KERNEL_H_
