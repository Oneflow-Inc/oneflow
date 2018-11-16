#include "oneflow/core/kernel/gather_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include <assert.h>
namespace oneflow {

namespace {

template<typename T, typename IndexT>
__global__ void LookupForwardGpu(const int64_t elem_cnt, const IndexT* indices, const T* in,
                                 int64_t in_blocks, int64_t in_rows, int64_t in_cols, T* out) {
  const int64_t num_indices = elem_cnt / in_blocks;
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int64_t out_block_size = num_indices * in_cols;
    const int64_t block_idx = i / out_block_size;
    const int64_t block_offset = i % out_block_size;
    const int64_t offset = block_offset / in_cols;
    const int64_t col = block_offset % in_cols;
    const int64_t idx = indices[offset];
    assert(idx >= 0 && idx < in_rows);
    out[i] = in[block_idx * in_rows * in_cols + idx * in_cols + col];
  }
}

template<typename T, typename IndexT>
__global__ void LookupBackwardGpu(const int64_t elem_cnt, const IndexT* indices, const T* out_diff,
                                  int64_t in_blocks, int64_t in_rows, int64_t in_cols, T* in_diff) {
  const int64_t num_indices = elem_cnt / in_blocks;
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int64_t out_block_size = num_indices * in_cols;
    const int64_t block_idx = i / out_block_size;
    const int64_t block_offset = i % out_block_size;
    const int64_t offset = block_offset / in_cols;
    const int64_t col = block_offset % in_cols;
    const int64_t idx = indices[offset];
    assert(idx >= 0 && idx < in_rows);
    gpu_atomic_add(in_diff + block_idx * in_rows * in_cols + idx * in_cols + col, out_diff[i]);
  }
}

}  // namespace

template<typename T, typename IndexT>
struct LookupKernelUtil<DeviceType::kGPU, T, IndexT> final {
  static void Forward(DeviceCtx* ctx, const IndexT* indices, int64_t num_indices, const T* in,
                      int64_t in_blocks, int64_t in_rows, int64_t in_cols, T* out);
  static void Backward(DeviceCtx* ctx, const IndexT* indices, int64_t num_indices,
                       const T* out_diff, int64_t in_blocks, int64_t in_rows, int64_t in_cols,
                       T* in_diff);
};

template<typename T, typename IndexT>
void LookupKernelUtil<DeviceType::kGPU, T, IndexT>::Forward(DeviceCtx* ctx, const IndexT* indices,
                                                            int64_t num_indices, const T* in,
                                                            int64_t in_blocks, int64_t in_rows,
                                                            int64_t in_cols, T* out) {
  const int64_t elem_cnt = in_blocks * num_indices * in_cols;
  LookupForwardGpu<T, IndexT>
      <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          elem_cnt, indices, in, in_blocks, in_rows, in_cols, out);
}

template<typename T, typename IndexT>
void LookupKernelUtil<DeviceType::kGPU, T, IndexT>::Backward(DeviceCtx* ctx, const IndexT* indices,
                                                             int64_t num_indices, const T* out_diff,
                                                             int64_t in_blocks, int64_t in_rows,
                                                             int64_t in_cols, T* in_diff) {
  const int64_t elem_cnt = in_blocks * num_indices * in_cols;
  LookupBackwardGpu<T, IndexT>
      <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          elem_cnt, indices, out_diff, in_blocks, in_rows, in_cols, in_diff);
}

#define MAKE_LOOK_UP_KERNEL_UTIL_ENTRY(in_type_pair, index_type_pair)                \
  template struct LookupKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(in_type_pair), \
                                   OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_LOOK_UP_KERNEL_UTIL_ENTRY, FLOATING_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ);
#undef MAKE_LOOK_UP_KERNEL_UTIL_ENTRY

}  // namespace oneflow
