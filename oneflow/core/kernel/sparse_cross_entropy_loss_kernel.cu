#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/kernel/sparse_cross_entropy_loss_kernel.h"

namespace oneflow {

namespace {

template<typename PredType, typename LabelType>
__global__ void SparseCrossEntropyLossForwardGpu(const int64_t instance_num,
                                                 const int64_t num_of_classes,
                                                 const PredType* prediction,
                                                 const LabelType* labels, PredType* loss) {
  CUDA_1D_KERNEL_LOOP(i, instance_num) {
    int64_t label = static_cast<int64_t>(labels[i]);
    assert(label >= 0);
    assert(label < num_of_classes);
    loss[i] = -SafeLog(prediction[i * num_of_classes + label]);
  }
}

template<typename PredType, typename LabelType>
__global__ void SparseCrossEntropyLossBackwardGpu(const int64_t instance_num,
                                                  const int64_t num_of_classes,
                                                  const PredType* prediction,
                                                  const LabelType* labels,
                                                  PredType* prediction_diff) {
  CUDA_1D_KERNEL_LOOP(i, instance_num) {
    int64_t label = static_cast<int64_t>(labels[i]);
    PredType prob = prediction[i * num_of_classes + label];
    prediction_diff[i * num_of_classes + label] = -1 / MaxWithLogThreshold(prob);
  }
}

__device__ half MaxWithLogThresholdHalf(const half x) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  half threshold = hexp2(__float2half(-14.0));
  if (__hgt(x, threshold)) { return x; }
  return threshold;
#else
  HALF_CHECK_FAILED;
  half ret;
  return ret;
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
}

__device__ half SafeLogHalf(const half x) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  return hlog(MaxWithLogThresholdHalf(x));
#else
  HALF_CHECK_FAILED;
  half ret;
  return ret;
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
}

template<typename LabelType>
__global__ void SparseCrossEntropyLossForwardHalfGpu(const int64_t instance_num,
                                                     const int64_t num_of_classes,
                                                     const half* prediction,
                                                     const LabelType* labels, half* loss) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, instance_num) {
    int64_t label = static_cast<int64_t>(labels[i]);
    assert(label >= 0);
    assert(label < num_of_classes);
    loss[i] = __hneg(SafeLogHalf(prediction[i * num_of_classes + label]));
  }
#else
  HALF_CHECK_FAILED;
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
}

template<typename LabelType>
__global__ void SparseCrossEntropyLossBackwardHalfGpu(const int64_t instance_num,
                                                      const int64_t num_of_classes,
                                                      const half* prediction,
                                                      const LabelType* labels,
                                                      half* prediction_diff) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, instance_num) {
    int64_t label = static_cast<int64_t>(labels[i]);
    prediction_diff[i * num_of_classes + label] =
        __hdiv(__float2half(-1.0), MaxWithLogThresholdHalf(prediction[i * num_of_classes + label]));
  }
#else
  HALF_CHECK_FAILED;
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
}

}  // namespace

template<typename PredType, typename LabelType>
struct SparseCrossEntropyLossKernelUtil<DeviceType::kGPU, PredType, LabelType> {
  static void Forward(DeviceCtx* ctx, const int64_t instance_num, const int64_t num_of_classes,
                      const PredType* prediction, const LabelType* labels, PredType* loss) {
    SparseCrossEntropyLossForwardGpu<PredType, LabelType>
        <<<BlocksNum4ThreadsNum(instance_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            instance_num, num_of_classes, prediction, labels, loss);
  }

  static void Backward(DeviceCtx* ctx, const int64_t instance_num, const int64_t num_of_classes,
                       const PredType* prediction, const LabelType* labels,
                       PredType* prediction_diff) {
    SparseCrossEntropyLossBackwardGpu<PredType, LabelType>
        <<<BlocksNum4ThreadsNum(instance_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            instance_num, num_of_classes, prediction, labels, prediction_diff);
  }
};

template<typename LabelType>
struct SparseCrossEntropyLossKernelUtil<DeviceType::kGPU, float16, LabelType> {
  static void Forward(DeviceCtx* ctx, const int64_t instance_num, const int64_t num_of_classes,
                      const float16* prediction, const LabelType* labels, float16* loss) {
    SparseCrossEntropyLossForwardHalfGpu<LabelType>
        <<<BlocksNum4ThreadsNum(instance_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            instance_num, num_of_classes, reinterpret_cast<const half*>(prediction), labels,
            reinterpret_cast<half*>(loss));
  }

  static void Backward(DeviceCtx* ctx, const int64_t instance_num, const int64_t num_of_classes,
                       const float16* prediction, const LabelType* labels,
                       float16* prediction_diff) {
    SparseCrossEntropyLossBackwardHalfGpu<LabelType>
        <<<BlocksNum4ThreadsNum(instance_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            instance_num, num_of_classes, reinterpret_cast<const half*>(prediction), labels,
            reinterpret_cast<half*>(prediction_diff));
  }
};

#define MAKE_ENTRY(data_type_pair, label_type_pair) \
  template struct SparseCrossEntropyLossKernelUtil< \
      DeviceType::kGPU, OF_PP_PAIR_FIRST(data_type_pair), OF_PP_PAIR_FIRST(label_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ)

}  // namespace oneflow
