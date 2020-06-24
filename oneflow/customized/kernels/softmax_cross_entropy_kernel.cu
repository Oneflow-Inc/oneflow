#include "oneflow/customized/kernels/softmax_cross_entropy_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {
namespace user_op {

namespace {

template<typename T>
__global__ void ComputeEntropyGpu(const int64_t num_instances, const int64_t num_classes,
                                  const T* x, const T* labels, T* y) {
  CUDA_1D_KERNEL_LOOP(i, num_instances * num_classes) {
    const int32_t row_id = i / num_classes;
    T label = labels[i];
    T prob = x[i];
    gpu_atomic_add(y + row_id, -label * SafeLog(prob));
  }
}

__global__ void ComputeEntropyGpuHalf(const int64_t num_instances, const int64_t num_classes,
                                      const half* x, const half* labels, half* y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && CUDA_VERSION >= 10000
  CUDA_1D_KERNEL_LOOP(i, num_instances * num_classes) {
    const int32_t row_id = i / num_classes;
    half label = labels[i];
    half prob = x[i];
    gpu_atomic_add(y + row_id, __hneg(__hmul(label, SafeLog<half>(prob))));
  }
#else
  printf("use half softmax cross entropy need nvcc arch >= 700 and cuda >= 10.0");
  assert(false);
#endif /* defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && CUDA_VERSION >= 10000 */
}

template<typename T>
__global__ void ComputeDiffWithSoftmaxGpu(const int64_t elem_cnt, const int64_t num_classes,
                                          const T* prob, const T* labels, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int32_t row_id = i / num_classes;
    dx[i] = dy[row_id] * (prob[i] - labels[i]);
  }
}

__global__ void ComputeDiffWithSoftmaxGpuHalf(const int64_t elem_cnt, const int64_t num_classes,
                                              const half* prob, const half* labels, const half* dy,
                                              half* dx) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int32_t row_id = i / num_classes;
    dx[i] = __hmul(dy[row_id], __hsub(prob[i], labels[i]));
  }
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

}  // namespace

template<typename T>
struct CrossEntropyKernelUtil<DeviceType::kGPU, T> {
  static void ComputeEntropy(DeviceCtx* ctx, const int64_t num_instances, const int64_t num_classes,
                             const T* x, const T* labels, T* y) {
    cudaMemset(y, 0, sizeof(T) * num_instances);
    ComputeEntropyGpu<<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0,
                        ctx->cuda_stream()>>>(num_instances, num_classes, x, labels, y);
  }

  static void ComputeDiffWithSoftmax(DeviceCtx* ctx, const int64_t elem_cnt,
                                     const int64_t num_classes, const T* prob, const T* labels,
                                     const T* dy, T* dx) {
    ComputeDiffWithSoftmaxGpu<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                ctx->cuda_stream()>>>(elem_cnt, num_classes, prob, labels, dy, dx);
  }
};

template<>
struct CrossEntropyKernelUtil<DeviceType::kGPU, float16> {
  static void ComputeEntropy(DeviceCtx* ctx, const int64_t num_instances, const int64_t num_classes,
                             const float16* x, const float16* labels, float16* y) {
    cudaMemset(y, 0, sizeof(float16) * num_instances);
    ComputeEntropyGpuHalf<<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0,
                            ctx->cuda_stream()>>>(
        num_instances, num_classes, reinterpret_cast<const half*>(x),
        reinterpret_cast<const half*>(labels), reinterpret_cast<half*>(y));
  }

  static void ComputeDiffWithSoftmax(DeviceCtx* ctx, const int64_t elem_cnt,
                                     const int64_t num_classes, const float16* prob,
                                     const float16* labels, const float16* dy, float16* dx) {
    ComputeDiffWithSoftmaxGpuHalf<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                    ctx->cuda_stream()>>>(
        elem_cnt, num_classes, reinterpret_cast<const half*>(prob),
        reinterpret_cast<const half*>(labels), reinterpret_cast<const half*>(dy),
        reinterpret_cast<half*>(dx));
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SOFTMAX_CROSS_ENTROPY_KERNEL,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kGPU),
                                 FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SOFTMAX_CROSS_ENTROPY_GRAD_KERNEL,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kGPU),
                                 FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ)
}  // namespace user_op
}  // namespace oneflow
