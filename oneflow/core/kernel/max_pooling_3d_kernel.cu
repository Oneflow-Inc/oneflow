#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/max_pooling_3d_kernel.h"

namespace oneflow {

template<typename T>
class MaxPooling3DKernelUtil<DeviceType::kGPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPooling3DKernelUtil);
  MaxPooling3DKernelUtil() = delete;

  static void Forward(const KernelCtx& ctx, const Blob* in_blob, Blob* out_blob,
                      Blob* idx_blob, const Pooling3DCtx& pooling_ctx) {
    TODO();
  }

  static void Backward(const KernelCtx& ctx, const Blob* out_diff_blob,
                       const Blob* idx_blob, Blob* in_diff_blob,
                       const Pooling3DCtx& pooling_ctx) {
    TODO();
  }
};

#define INSTANTIATE_MAX_POOLING_3D_KERNEL_UTIL(type_cpp, type_proto) \
  template class MaxPooling3DKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_MAX_POOLING_3D_KERNEL_UTIL,
                     ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
