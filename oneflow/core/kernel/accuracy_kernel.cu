
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/kernel/accuracy_kernel.h"
#include <cub/cub.cuh>
namespace oneflow {

namespace {

template<typename PredType, typename LabelType>
__global__ void AccuracyComputeKernel(const int32_t n, const int32_t d, const int32_t top_k,
                                      const PredType* prediction, const LabelType* label,
                                      PredType* accuracy) {
  typedef cub::BlockReduce<int32_t, kCudaThreadsNumPerBlock> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int32_t correct = 0;
  for (int32_t i = blockIdx.x; i < n; i += gridDim.x) {
    const LabelType label_i = label[i];
    const PredType pred_i = prediction[i * d + label_i];
    int32_t ngt = 0;
    for (int32_t j = threadIdx.x; j < d; j += blockDim.x) {
      const PredType pred_tmp = prediction[i * d + j];
      if ((pred_tmp > pred_i) || (pred_tmp == pred_i && j <= label_i)) {
        if (++ngt > top_k) { break; }
      }
    }
    ngt = BlockReduce(temp_storage).Sum(ngt);
    if (ngt <= top_k) { ++correct; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { gpu_atomic_add(accuracy, static_cast<PredType>(correct)); }
}

}  // namespace

template<typename PredType, typename LabelType>
struct AccuracyKernelUtil<DeviceType::kGPU, PredType, LabelType> {
  static void Forward(DeviceCtx* ctx, const int32_t n, const int32_t d, int32_t top_k,
                      const PredType* prediction, const LabelType* label, PredType* accuracy) {
    AccuracyComputeKernel<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                            ctx->cuda_stream()>>>(n, d, top_k, prediction, label, accuracy);
  };
};

#define MAKE_ENTRY(data_type_pair, label_type_pair)                                      \
  template struct AccuracyKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(data_type_pair), \
                                     OF_PP_PAIR_FIRST(label_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)
}  // namespace oneflow
