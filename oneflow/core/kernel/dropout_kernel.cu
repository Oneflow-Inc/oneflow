#include "oneflow/core/kernel/dropout_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void MaskAndScaleGpu(const int64_t n, float threshold, float scale, const T* x,
                                const float* random_mask, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = x[i] * (random_mask[i] > threshold) * scale; }
}

template<>
__global__ void MaskAndScaleGpu<half>(const int64_t n, float threshold, float scale, const half* x,
                                      const float* random_mask, half* y) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  half h_scale = __float2half(scale);
  CUDA_1D_KERNEL_LOOP(i, n) {
    half one_or_zero = random_mask[i] > threshold;
    y[i] = __hmul(__hmul(x[i], one_or_zero), h_scale);
  }
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

}  // namespace

template<typename T>
struct DropoutKernelUtil<DeviceType::kGPU, T, typename std::enable_if<!IsFloat16<T>::value>::type>
    final {
  static void MaskAndScale(DeviceCtx* ctx, const int64_t n, float threshold, float scale,
                           const T* x, const float* random_mask, T* y) {
    MaskAndScaleGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, threshold, scale, x, random_mask, y);
  }
};

template<typename T>
struct DropoutKernelUtil<DeviceType::kGPU, T, typename std::enable_if<IsFloat16<T>::value>::type>
    final {
  static void MaskAndScale(DeviceCtx* ctx, const int64_t n, float threshold, float scale,
                           const T* x, const float* random_mask, T* y) {
    MaskAndScaleGpu<half>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            n, threshold, scale, reinterpret_cast<const half*>(x), random_mask,
            reinterpret_cast<half*>(y));
  }
};

#define INITIATE_DROPOUT_KERNEL_UTIL_GPU(T, type_proto) \
  template struct DropoutKernelUtil<DeviceType::kGPU, T>;
OF_PP_FOR_EACH_TUPLE(INITIATE_DROPOUT_KERNEL_UTIL_GPU,
                     ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ);
#undef INITIATE_DROPOUT_KERNEL_UTIL_GPU

}  // namespace oneflow
