#include "oneflow/core/kernel/gather_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include <assert.h>

namespace oneflow {

namespace {

template<typename K>
__device__ int64_t get_in_offset(int64_t out_offset, const K* indices, int64_t num_indices,
                                 int64_t gather_dim_size, int64_t inner_dim_size) {
  const int64_t outer_dim_elem_cnt = num_indices * inner_dim_size;
  const int64_t outer_idx = out_offset / outer_dim_elem_cnt;
  const int64_t indices_idx = out_offset % outer_dim_elem_cnt / inner_dim_size;
  const int64_t inner_idx = out_offset % inner_dim_size;
  const int64_t idx = indices[indices_idx];
  assert(idx >= 0 && idx < gather_dim_size);
  return outer_idx * gather_dim_size * inner_dim_size + idx * inner_dim_size + inner_idx;
}

template<typename T, typename K>
__global__ void GatherForwardGpu(int64_t elem_cnt, const K* indices, int64_t num_indices,
                                 const T* in, int64_t gather_dim_size, int64_t inner_dim_size,
                                 T* out) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    out[i] = in[get_in_offset<K>(i, indices, num_indices, gather_dim_size, inner_dim_size)];
  }
}

template<typename T, typename K>
__global__ void GatherBackwardGpu(int64_t elem_cnt, const K* indices, int64_t num_indices,
                                  const T* out_diff, int64_t gather_dim_size,
                                  int64_t inner_dim_size, T* in_diff) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    gpu_atomic_add(
        in_diff + get_in_offset<K>(i, indices, num_indices, gather_dim_size, inner_dim_size),
        out_diff[i]);
  }
}

}  // namespace

template<typename T, typename K>
struct GatherKernelUtil<DeviceType::kGPU, T, K> final {
  static void Forward(DeviceCtx* ctx, const K* indices, int64_t num_indices, const T* in,
                      const Shape& flat_in_shape, T* out);
  static void Backward(DeviceCtx* ctx, const K* indices, int64_t num_indices, const T* out_diff,
                       const Shape& flat_in_shape, T* in_diff);
};

template<typename T, typename K>
void GatherKernelUtil<DeviceType::kGPU, T, K>::Forward(DeviceCtx* ctx, const K* indices,
                                                       int64_t num_indices, const T* in,
                                                       const Shape& flat_in_shape, T* out) {
  const int64_t elem_cnt = flat_in_shape.At(0) * num_indices * flat_in_shape.At(2);
  GatherForwardGpu<T, K>
      <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          elem_cnt, indices, num_indices, in, flat_in_shape.At(1), flat_in_shape.At(2), out);
}

template<typename T, typename K>
void GatherKernelUtil<DeviceType::kGPU, T, K>::Backward(DeviceCtx* ctx, const K* indices,
                                                        int64_t num_indices, const T* out_diff,
                                                        const Shape& flat_in_shape, T* in_diff) {
  const int64_t elem_cnt = flat_in_shape.At(0) * num_indices * flat_in_shape.At(2);
  GatherBackwardGpu<T, K>
      <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          elem_cnt, indices, num_indices, out_diff, flat_in_shape.At(1), flat_in_shape.At(2),
          in_diff);
}

#define MAKE_GATHER_KERNEL_UTIL_ENTRY(in_type_pair, index_type_pair)                 \
  template struct GatherKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(in_type_pair), \
                                   OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_GATHER_KERNEL_UTIL_ENTRY, FLOATING_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ);
#undef MAKE_GATHER_KERNEL_UTIL_ENTRY

}  // namespace oneflow
