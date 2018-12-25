#ifndef ONEFLOW_CORE_KERNEL_NORM_KERNEL_CUH_
#define ONEFLOW_CORE_KERNEL_NORM_KERNEL_CUH_

namespace oneflow {
template<typename T>
__host__ __device__ T L1NormInDiff(T out_diff, T in) {
  return out_diff * ((in >= 0) - (in <= 0));
}
}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NORM_KERNEL_CUH_
