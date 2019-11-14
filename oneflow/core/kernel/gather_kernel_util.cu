#include "oneflow/core/kernel/gather_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include <assert.h>

namespace oneflow {

namespace {

template<typename K, typename IDX>
__device__ IDX GetInOffset(const IDX out_offset, const K* indices, const IDX num_indices,
                           const IDX gather_dim_size, const IDX inner_dim_size, const IDX offset) {
  const IDX outer_dim_elem_cnt = num_indices * inner_dim_size;
  const IDX outer_idx = out_offset / outer_dim_elem_cnt;
  const IDX indices_idx = out_offset % outer_dim_elem_cnt / inner_dim_size;
  const IDX inner_idx = out_offset % inner_dim_size;
  assert(indices[indices_idx] >= 0);
  const IDX idx = indices[indices_idx] - offset;
  if (idx >= 0 && idx < gather_dim_size) {
    return outer_idx * gather_dim_size * inner_dim_size + idx * inner_dim_size + inner_idx;
  } else {
    return -1;
  }
}

template<typename T, typename K, typename IDX>
__global__ void GatherForwardGpu(const IDX elem_cnt, const K* indices, const IDX num_indices,
                                 const T* in, const IDX gather_dim_size, const IDX inner_dim_size,
                                 T* out, const IDX offset) {
  CUDA_1D_KERNEL_LOOP_T(IDX, i, elem_cnt) {
    const IDX in_offset =
        GetInOffset<K, IDX>(i, indices, num_indices, gather_dim_size, inner_dim_size, offset);
    if (in_offset < 0) {
      out[i] = 0;
    } else {
      out[i] = in[in_offset];
    }
  }
}

template<typename T, typename K, typename IDX>
__global__ void GatherBackwardGpu(const IDX elem_cnt, const K* indices, const IDX num_indices,
                                  const T* out_diff, const IDX gather_dim_size,
                                  const IDX inner_dim_size, T* in_diff, const IDX offset) {
  CUDA_1D_KERNEL_LOOP_T(IDX, i, elem_cnt) {
    const T diff_val = out_diff[i];
    if (diff_val != static_cast<T>(0)) {
      const int64_t in_offset =
          GetInOffset<K, IDX>(i, indices, num_indices, gather_dim_size, inner_dim_size, offset);
      if (in_offset >= 0) { gpu_atomic_add(in_diff + in_offset, diff_val); }
    }
  }
}

bool IsSafeUseIndex32(const Shape& flat_in_shape, const int64_t num_indices) {
  const int64_t in_elem_cnt = flat_in_shape.elem_cnt();
  const int64_t out_elem_cnt = flat_in_shape.At(0) * num_indices * flat_in_shape.At(2);
  return std::max(out_elem_cnt, in_elem_cnt) < GetMaxVal<int32_t>() / 2;
}

}  // namespace

template<typename T, typename K>
struct GatherKernelUtilImpl<DeviceType::kGPU, T, K> final {
  static void Forward(DeviceCtx* ctx, const K* indices, int64_t num_indices, const T* in,
                      const Shape& flat_in_shape, T* out, const int64_t offset) {
    const int64_t out_elem_cnt = flat_in_shape.At(0) * num_indices * flat_in_shape.At(2);
    if (IsSafeUseIndex32(flat_in_shape, num_indices)) {
      GatherForwardGpu<T, K, int32_t>
          <<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              out_elem_cnt, indices, num_indices, in, flat_in_shape.At(1), flat_in_shape.At(2), out,
              offset);
    } else {
      GatherForwardGpu<T, K, int64_t>
          <<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              out_elem_cnt, indices, num_indices, in, flat_in_shape.At(1), flat_in_shape.At(2), out,
              offset);
    }
  }
  static void Backward(DeviceCtx* ctx, const K* indices, int64_t num_indices, const T* out_diff,
                       const Shape& flat_in_shape, T* in_diff, const int64_t offset) {
    const int64_t elem_cnt = flat_in_shape.At(0) * num_indices * flat_in_shape.At(2);
    if (IsSafeUseIndex32(flat_in_shape, num_indices)) {
      GatherBackwardGpu<T, K, int32_t>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              elem_cnt, indices, num_indices, out_diff, flat_in_shape.At(1), flat_in_shape.At(2),
              in_diff, offset);
    } else {
      GatherBackwardGpu<T, K, int64_t>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              elem_cnt, indices, num_indices, out_diff, flat_in_shape.At(1), flat_in_shape.At(2),
              in_diff, offset);
    }
  }
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && CUDA_VERSION >= 10000
template<typename K>
struct GatherKernelUtilImpl<DeviceType::kGPU, float16, K> final {
  static void Forward(DeviceCtx* ctx, const K* indices, int64_t num_indices, const float16* in,
                      const Shape& flat_in_shape, float16* out, const int64_t offset) {
    GatherKernelUtilImpl<DeviceType::kGPU, half, K>::Forward(
        ctx, indices, num_indices, reinterpret_cast<const half*>(in), flat_in_shape,
        reinterpret_cast<half*>(out), offset);
  }
  static void Backward(DeviceCtx* ctx, const K* indices, int64_t num_indices,
                       const float16* out_diff, const Shape& flat_in_shape, float16* in_diff,
                       const int64_t offset) {
    GatherKernelUtilImpl<DeviceType::kGPU, half, K>::Backward(
        ctx, indices, num_indices, reinterpret_cast<const half*>(out_diff), flat_in_shape,
        reinterpret_cast<half*>(in_diff), offset);
  }
};
#endif

#define INITIATE_GATHER_KERNEL_UTIL_GPU_IMPL(in_type_pair, index_type_pair)              \
  template struct GatherKernelUtilImpl<DeviceType::kGPU, OF_PP_PAIR_FIRST(in_type_pair), \
                                       OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_GATHER_KERNEL_UTIL_GPU_IMPL,
                                 FLOATING_DATA_TYPE_SEQ
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && CUDA_VERSION >= 10000
                                     FLOAT16_DATA_TYPE_SEQ
#endif
                                 ,
                                 INDEX_DATA_TYPE_SEQ);
#undef INITIATE_GATHER_KERNEL_UTIL_GPU_IMPL

}  // namespace oneflow
