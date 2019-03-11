#include "oneflow/core/kernel/loss_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void LossReductionAssign(T* reduction, T val) {
  *reduction = val;
}

template<typename T>
__global__ void LossReductionAssignN(T* reduction, const T* weight, int64_t n) {
  *reduction = (*weight) * n;
}

template<typename T>
__global__ void LossReductionCountNonZero(T* reduction, const T* weight, int64_t n) {
  T ret = 0.0;
  for (int32_t i = 0; i < n; i++) {
    if (weight[i] > 0) { ret += 1.0; }
  }
  *reduction = ret;
}

__global__ void LossReductionAssignNHalf(half* reduction, const half* weight, int64_t n) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  *reduction = __hmul((*weight), __float2half(n * 1.0));
#else
  HALF_CHECK_FAILED;
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
}

__global__ void LossReductionCountNonZeroHalf(half* reduction, const half* weight, int64_t n) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  half ret = __float2half(0.0);
  for (int32_t i = 0; i < n; i++) {
    if (__hgt(weight[i], __float2half(0.0))) { ret = __hadd(ret, __float2half(1.0)); }
  }
  *reduction = ret;
#else
  HALF_CHECK_FAILED;
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
}

}  // namespace

template<typename T>
struct LossKernelUtil<DeviceType::kGPU, T, typename std::enable_if<!IsFloat16<T>::value>::type> {
  static void ComputeReductionCoefficient(DeviceCtx* ctx, int64_t data_num, int64_t weight_length,
                                          const T* weight, T* reduction, LossReductionType type) {
    switch (type) {
      case kSumOverOne: {
        LossReductionAssign<T><<<1, 1, 0, ctx->cuda_stream()>>>(reduction, 1.0);
        break;
      }
      case kSumOverWeight: {
        if (weight_length == data_num) {
          NewKernelUtil<DeviceType::kGPU, T>::Sum(ctx, weight_length, weight, reduction);
        } else if (weight_length == 1) {
          LossReductionAssignN<T><<<1, 1, 0, ctx->cuda_stream()>>>(reduction, weight, data_num);
        } else {
          UNIMPLEMENTED();
        }
        break;
      }
      case kSumOverN: {
        LossReductionAssign<T><<<1, 1, 0, ctx->cuda_stream()>>>(reduction, data_num * 1.0);
        break;
      }
      case kSumOverNonZeroWeight: {
        if (weight_length == data_num) {
          LossReductionCountNonZero<T>
              <<<1, 1, 0, ctx->cuda_stream()>>>(reduction, weight, data_num);
        } else if (weight_length == 1) {
          LossReductionAssign<T><<<1, 1, 0, ctx->cuda_stream()>>>(reduction, data_num * 1.0);
        } else {
          UNIMPLEMENTED();
        }
        break;
      }
      default: UNIMPLEMENTED();
    }
  }
};

template<typename T>
struct LossKernelUtil<DeviceType::kGPU, T, typename std::enable_if<IsFloat16<T>::value>::type> {
  static void ComputeReductionCoefficient(DeviceCtx* ctx, int64_t data_num, int64_t weight_length,
                                          const T* weight, T* reduction, LossReductionType type) {
    const half* weight_h = reinterpret_cast<const half*>(weight);
    half* reduction_h = reinterpret_cast<half*>(reduction);
    switch (type) {
      case kSumOverOne: {
        T hone = oneflow_cast<T>(1.0);
        LossReductionAssign<half>
            <<<1, 1, 0, ctx->cuda_stream()>>>(reduction_h, *(reinterpret_cast<half*>(&hone)));
        break;
      }
      case kSumOverWeight: {
        if (weight_length == data_num) {
          NewKernelUtil<DeviceType::kGPU, T>::Sum(ctx, weight_length, weight, reduction);
        } else if (weight_length == 1) {
          LossReductionAssignNHalf<<<1, 1, 0, ctx->cuda_stream()>>>(reduction_h, weight_h,
                                                                    data_num);
        } else {
          UNIMPLEMENTED();
        }
        break;
      }
      case kSumOverN: {
        T ret = oneflow_cast<T>(data_num * 1.0);
        LossReductionAssign<half>
            <<<1, 1, 0, ctx->cuda_stream()>>>(reduction_h, *(reinterpret_cast<half*>(&ret)));
        break;
      }
      case kSumOverNonZeroWeight: {
        if (weight_length == data_num) {
          LossReductionCountNonZeroHalf<<<1, 1, 0, ctx->cuda_stream()>>>(reduction_h, weight_h,
                                                                         data_num);
        } else if (weight_length == 1) {
          T ret = oneflow_cast<T>(data_num * 1.0);
          LossReductionAssign<half>
              <<<1, 1, 0, ctx->cuda_stream()>>>(reduction_h, *(reinterpret_cast<half*>(&ret)));
        } else {
          UNIMPLEMENTED();
        }
        break;
      }
      default: UNIMPLEMENTED();
    }
  }
};

#define MAKE_LOSS_KERNEL_UTIL_ENTRY(type_cpp, type_proto) \
  template struct LossKernelUtil<DeviceType::kGPU, type_cpp>;

OF_PP_FOR_EACH_TUPLE(MAKE_LOSS_KERNEL_UTIL_ENTRY, FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ)

}  // namespace oneflow
