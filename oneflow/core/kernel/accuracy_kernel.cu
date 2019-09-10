
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/kernel/accuracy_kernel.h"
#include <cub/cub.cuh>
namespace oneflow {

namespace {
template<typename PredType>
__global__ void AccuracySetZeroKernel(PredType* accuracy) {
  *accuracy = 0;
}

template<typename PredType, typename LabelType>
__global__ void AccuracyComputeKernel(const int32_t N, const int32_t D, const int32_t top_k,
                                      const PredType* Xdata, const LabelType* labelData,
                                      const PredType* weight, PredType* accuracy) {
  typedef cub::BlockReduce<int32_t, kCudaThreadsNumPerBlock> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  PredType correct = 0;
  for (int32_t row = blockIdx.x; row < N; row += gridDim.x) {
    const LabelType label = labelData[row];
    const PredType label_pred = Xdata[row * D + label];
    int32_t ngt = 0;
    for (int32_t col = threadIdx.x; col < D; col += blockDim.x) {
      const PredType pred = Xdata[row * D + col];
      if (pred > label_pred || (pred == label_pred && col <= label)) { ++ngt; }
    }
    ngt = BlockReduce(temp_storage).Sum(ngt);
    if (ngt <= top_k) { correct += weight ? weight[row] : GetOneVal<PredType>(); }
    __syncthreads();
  }
  if (threadIdx.x == 0) { gpu_atomic_add(accuracy, correct); }
}
}  // namespace

template<typename PredType, typename LabelType>
struct AccuracyKernelUtil<DeviceType::kGPU, PredType, LabelType> {
  static void Forward(DeviceCtx* ctx, const int32_t N, const int32_t D, int32_t top_k,
                      const PredType* XData, const LabelType* labelData, const PredType* weight,
                      PredType* accuracyData) {
    AccuracySetZeroKernel<<<1, 1, 0, ctx->cuda_stream()>>>(accuracyData);
    AccuracyComputeKernel<<<BlocksNum4ThreadsNum(N), kCudaThreadsNumPerBlock, 0,
                            ctx->cuda_stream()>>>(N, D, top_k, XData, labelData, weight,
                                                  accuracyData);
  };
};
#define MAKE_ENTRY(data_type_pair, label_type_pair)                                      \
  template struct AccuracyKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(data_type_pair), \
                                     OF_PP_PAIR_FIRST(label_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)
}  // namespace oneflow
