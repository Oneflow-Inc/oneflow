#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/normal_model_update_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void CmptClipRatioByGlobalNormGpu(const T* global_norm_ptr, T clip_norm, T* ratio_ptr) {
  *ratio_ptr = clip_norm / max(*global_norm_ptr, clip_norm);
}

}  // namespace

template<typename T>
class NormalMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void CmptClipRatioByGlobalNorm(DeviceCtx* ctx, const T* global_norm_ptr, T clip_norm,
                                        T* ratio_ptr) {
    CmptClipRatioByGlobalNormGpu<T>
        <<<1, 1, 0, ctx->cuda_stream()>>>(global_norm_ptr, clip_norm, ratio_ptr);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class NormalMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
