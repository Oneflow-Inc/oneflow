#include "oneflow/core/kernel/one_hot_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include <assert.h>

namespace oneflow {

namespace {

template<typename T, typename K>
__global__ void OneHotEncodeGpu(int64_t elem_cnt, const K* indices, int64_t depth, T* out) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int64_t row = i / depth;
    const int64_t col = i % depth;
    const int64_t idx = indices[row];
    assert(idx >= 0 && idx < depth);
    out[i] = (idx == col);
  }
}

}  // namespace

template<typename T, typename K>
struct OneHotKernelUtil<DeviceType::kGPU, T, K> final {
  static void Encode(DeviceCtx* ctx, const K* indices, int64_t num_indices, int64_t depth, T* out);
};

template<typename T, typename K>
void OneHotKernelUtil<DeviceType::kGPU, T, K>::Encode(DeviceCtx* ctx, const K* indices,
                                                      int64_t num_indices, int64_t depth, T* out) {
  const int64_t elem_cnt = num_indices * depth;
  OneHotEncodeGpu<T, K>
      <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          elem_cnt, indices, depth, out);
}

#define INSTANTIATE_ONE_HOT_KERNEL_UTIL_GPU(data_type_pair, index_type_pair)           \
  template struct OneHotKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(data_type_pair), \
                                   OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_ONE_HOT_KERNEL_UTIL_GPU, ARITHMETIC_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ);
#undef INSTANTIATE_ONE_HOT_KERNEL_UTIL_GPU

}  // namespace oneflow
