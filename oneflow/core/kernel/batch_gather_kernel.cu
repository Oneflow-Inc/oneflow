#include "oneflow/core/kernel/batch_gather_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include <assert.h>

namespace oneflow {

namespace {

template<typename K>
__device__ int64_t get_in_offset(const int64_t out_offset, const K* indices,
                                 const int64_t batch_num, const int64_t indices_num,
                                 const int64_t instance_dim, const int64_t gather_dim_size) {
  const int64_t batch_idx = out_offset / (indices_num * instance_dim);
  const int64_t indices_idx = (out_offset % batch_num) / instance_dim;
  const int64_t inner_idx = out_offset % instance_dim;
  const int64_t idx = indices[indices_idx];
  assert(idx >= 0 && idx < gather_dim_size);
  return batch_idx * gather_dim_size * instance_dim + idx * instance_dim + inner_idx;
}

template<typename T, typename K>
__global__ void BatchGatherForwardGpu(const int64_t elem_cnt, const T* in, const K* indices,
                                      const int64_t batch_num, const int64_t indices_num,
                                      const int64_t instance_dim, const int64_t gather_dim_size,
                                      T* out) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    out[i] =
        in[get_in_offset<K>(i, indices, batch_num, indices_num, instance_dim, gather_dim_size)];
  }
}

template<typename T, typename K>
__global__ void BatchGatherBackwardGpu(const int64_t elem_cnt, const T* out_diff, const K* indices,
                                       const int64_t batch_num, const int64_t indices_num,
                                       const int64_t instance_dim, const int64_t gather_dim_size,
                                       T* in_diff) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    gpu_atomic_add(
        in_diff
            + get_in_offset<K>(i, indices, batch_num, indices_num, instance_dim, gather_dim_size),
        out_diff[i]);
  }
}

}  // namespace

template<typename T, typename K>
struct BatchGatherKernelUtil<DeviceType::kGPU, T, K> final {
  static void Forward(DeviceCtx* ctx, const T* in, const K* indices, const Shape& flat_out_shape,
                      const int64_t gather_dim_size, T* out);
  static void Backward(DeviceCtx* ctx, const T* out_diff, const K* indices,
                       const Shape& flat_out_diff_shape, const int64_t gather_dim_size, T* in_diff);
};

template<typename T, typename K>
void BatchGatherKernelUtil<DeviceType::kGPU, T, K>::Forward(DeviceCtx* ctx, const T* in,
                                                            const K* indices,
                                                            const Shape& flat_out_shape,
                                                            const int64_t gather_dim_size, T* out) {
  const int64_t batch_num = flat_out_shape.At(0);
  const int64_t indices_num = flat_out_shape.At(1);
  const int64_t instance_dim = flat_out_shape.At(2);
  const int64_t elem_cnt = batch_num * indices_num * instance_dim;
  BatchGatherForwardGpu<T, K>
      <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          elem_cnt, in, indices, batch_num, indices_num, instance_dim, gather_dim_size, out);
}

template<typename T, typename K>
void BatchGatherKernelUtil<DeviceType::kGPU, T, K>::Backward(DeviceCtx* ctx, const T* out_diff,
                                                             const K* indices,
                                                             const Shape& flat_out_diff_shape,
                                                             const int64_t gather_dim_size,
                                                             T* in_diff) {
  const int64_t batch_num = flat_out_diff_shape.At(0);
  const int64_t indices_num = flat_out_diff_shape.At(1);
  const int64_t instance_dim = flat_out_diff_shape.At(2);
  const int64_t elem_cnt = batch_num * indices_num * instance_dim;
  BatchGatherBackwardGpu<T, K>
      <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          elem_cnt, out_diff, indices, batch_num, indices_num, instance_dim, gather_dim_size,
          in_diff);
}

#define MAKE_BATCH_GATHER_KERNEL_UTIL_ENTRY(in_type_pair, index_type_pair)                \
  template struct BatchGatherKernelUtil<DeviceType::kGPU, OF_PP_PAIR_FIRST(in_type_pair), \
                                        OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_BATCH_GATHER_KERNEL_UTIL_ENTRY, FLOATING_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ);
#undef MAKE_BATCH_GATHER_KERNEL_UTIL_ENTRY

}  // namespace oneflow
