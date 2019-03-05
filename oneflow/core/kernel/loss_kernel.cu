#include "oneflow/core/kernel/loss_kernel.h"

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

}  // namespace

template<typename T>
struct LossKernelUtil<DeviceType::kGPU, T> {
  static void ComputeReductionCoefficient(DeviceCtx* ctx, int64_t data_num, int64_t weight_length,
                                          const T* weight, T* reduction, LossReductionType type) {
    switch (type) {
      case kSumOverOne: {
        LossReductionAssign<T><<<1, 1, 0, ctx->cuda_stream()>>>(reduction, 1.0);
        break;
      }
      case kSumOverWeight: {
        if (weight_length == data_num) {
          KernelUtil<DeviceType::kGPU, T>::Sum(ctx, weight_length, weight, reduction, nullptr, 0);
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

#define MAKE_LOSS_KERNEL_UTIL_ENTRY(type_cpp, type_proto) \
  template struct LossKernelUtil<DeviceType::kGPU, type_cpp>;

OF_PP_FOR_EACH_TUPLE(MAKE_LOSS_KERNEL_UTIL_ENTRY, FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ)

}  // namespace oneflow
