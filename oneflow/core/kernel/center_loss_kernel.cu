#include "oneflow/core/kernel/center_loss_kernel.h"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

template<typename PredType, typename LabelType>
__global__ void ForwardGpu(const PredType* prediction, const LabelType* label,
                           const int32_t num_classes, const int32_t dim, const int32_t num_labels,
                           const float alpha, PredType* piece_centers, PredType* centers,
                           PredType* loss, PredType* prediction_diff) {
  using BlockReduce = cub::BlockReduce<PredType, kCudaThreadsNumPerBlock>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int32_t i = blockIdx.x; i < num_labels; i += gridDim.x) {
    // Lookup
    for (int32_t j = threadIdx.x; j < dim; j += blockIdx.x) {
      assert(label[i] >= 0 && label[i] < num_classes);
      piece_centers[i * dim + j] = centers[label[i] * dim + j];
    }
    for (int32_t j = threadIdx.x; j < dim; j += blockIdx.x) {
      int64_t index = i * dim + j;
      // Forward
      PredType diff = prediction[index] - piece_centers[index];
      PredType loss_reduce_sum = BlockReduce(temp_storage).Sum(diff);
      if (threadIdx.x == 0) { loss[i] = loss_reduce_sum; }
      // Update Centers
      PredType center_diff = (1 - alpha) * (piece_centers[index] - prediction[index]);
      centers[index] -= center_diff;
      // Backward
      prediction_diff[index] = prediction[index] - piece_centers[index];
    }
  }
}

}  // namespace

template<typename PredType, typename LabelType>
struct CenterLossKernelUtil<DeviceType::kGPU, PredType, LabelType> {
  static void Forward(DeviceCtx* ctx, const PredType* prediction, const LabelType* label,
                      const int32_t num_classes, const int32_t dim, const int32_t num_labels,
                      const float alpha, PredType* piece_centers, PredType* centers, PredType* loss,
                      PredType* prediction_diff) {
    ForwardGpu<PredType, LabelType>
        <<<std::min(num_labels, kCudaMaxBlocksNum), kCudaThreadsNumPerBlock, 0,
           ctx->cuda_stream()>>>(prediction, label, num_classes, dim, num_labels, alpha,
                                 piece_centers, centers, loss, prediction_diff);
  }
};

#define MAKE_ENTRY(data_type_pair, label_type_pair)                                        \
  template struct CenterLossKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(data_type_pair), \
                                       OF_PP_PAIR_FIRST(label_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
