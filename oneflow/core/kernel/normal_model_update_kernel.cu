#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/normal_model_update_kernel.h"
#include "oneflow/core/kernel/normal_model_update_kernel.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void ClipByGlobalNormGpu(int64_t n, const T clip_radio, const T* global_norm,
                                    T* model_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    model_diff[i] = model_diff[i] * clip_radio / max(*global_norm, clip_radio);
  }
}

}  // namespace

template<typename T>
class NormalMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void ClipByGlobalNorm(DeviceCtx* ctx, const ClipByGlobalNorm& conf,
                               const T* batch_instance_num_ptr,
                               std::function<Blob*(const std::string&)> BnInOp2Blob) {
    int64_t n = BnInOp2Blob("model_diff")->shape().elem_cnt();
    T* model_diff = BnInOp2Blob("model_diff")->mut_dptr<T>();
    T* global_norm = BnInOp2Blob("global_norm")->mut_dptr<T>();
    if (conf.has_global_norm()) {
      *global_norm = static_cast<T>(conf.global_norm());
    } else {
      Memset<DeviceType::kGPU>(ctx, global_norm, 0,
                               BnInOp2Blob("global_norm")->ByteSizeOfDataContentField());
      KernelUtil<DeviceType::kGPU, T>::Dot(ctx, n, model_diff, 1, model_diff, 1, global_norm);
      KernelUtil<DeviceType::kGPU, T>::Sqrt(ctx, n, global_norm, global_norm);
      KernelUtil<DeviceType::kGPU, T>::Div(ctx, n, global_norm, batch_instance_num_ptr);
    }
    ClipByGlobalNormGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, static_cast<T>(conf.clip_radio()), global_norm, model_diff);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class NormalMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
