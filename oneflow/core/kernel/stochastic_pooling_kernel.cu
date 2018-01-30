#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/stochastic_pooling_kernel.h"

namespace oneflow {

template<typename T>
class StochasticPoolingKernelUtil<DeviceType::kGPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StochasticPoolingKernelUtil);
  StochasticPoolingKernelUtil() = delete;

  static void PoolingForward(const KernelCtx& ctx, const Blob* in_blob,
                             Blob* out_blob, Blob* mask_blob,
                             const StochasticPoolingOpConf& pooling_conf) {
    TODO();
  }

  static void PoolingBackward(const KernelCtx& ctx, const Blob* out_diff_blob,
                              const Blob* mask_blob, Blob* in_diff_blob,
                              const StochasticPoolingOpConf& pooling_conf) {
    TODO();
  }
};  // namespace oneflow

#define INSTANTIATE_STOCHASTIC_POOLING_KERNEL_UTIL(type_cpp, type_proto) \
  template class StochasticPoolingKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_STOCHASTIC_POOLING_KERNEL_UTIL,
                     ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
