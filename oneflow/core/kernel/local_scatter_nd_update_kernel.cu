#include "oneflow/core/kernel/local_scatter_nd_update_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<typename K>
__device__ int64_t GetOffset(const int64_t* shape_ptr, const int64_t shape_dim, const K* index_ptr,
                             const int64_t index_dim, const int64_t block_size,
                             const int64_t update_idx) {
  int64_t offset = 0;
  FOR_RANGE(int64_t, i, 0, index_dim) {
    int64_t stride = 1;
    FOR_RANGE(int64_t, j, i + 1, shape_dim) { stride *= shape_ptr[j]; }
    offset += index_ptr[i] * stride;
  }
  return offset * block_size + (update_idx % block_size);
}

template<typename T, typename K>
__global__ void GpuForward(const int64_t elem_cnt, const T* updates_ptr, const int64_t* shape_ptr,
                           const int64_t shape_dim, const K* indices_ptr, const int64_t index_dim,
                           const int64_t block_size, T* out_ptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    out_ptr[GetOffset(shape_ptr, shape_dim, indices_ptr + (i / block_size) * index_dim, index_dim,
                      block_size, i)] = 0;
  }
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    gpu_atomic_add(
        out_ptr
            + GetOffset<K>(shape_ptr, shape_dim, indices_ptr + (i / block_size) * index_dim,
                           index_dim, block_size, i),
        updates_ptr[i]);
  }
}

template<typename T, typename K>
__global__ void GpuBackward(const int64_t elem_cnt, const T* out_diff_ptr, const int64_t* shape_ptr,
                            const int64_t shape_dim, const K* indices_ptr, const int64_t index_dim,
                            const int64_t block_size, T* updates_diff_ptr, T* in_diff_ptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const K* index_ptr = indices_ptr + (i / block_size) * index_dim;
    updates_diff_ptr[i] =
        out_diff_ptr[GetOffset(shape_ptr, shape_dim, index_ptr, index_dim, block_size, i)];
    in_diff_ptr[GetOffset(shape_ptr, shape_dim, index_ptr, index_dim, block_size, i)] = 0;
  }
}

}  // namespace

template<typename T, typename K>
struct LocalScatterNdUpdateKernelUtil<DeviceType::kGPU, T, K> {
  static void Forward(DeviceCtx* ctx, int64_t* shape_ptr, const Blob* indices_blob,
                      const Blob* updates_blob, const int64_t num_updates, const int64_t block_size,
                      Blob* out_blob) {
    CHECK_NOTNULL(shape_ptr);
    const int64_t shape_dim = out_blob->shape().NumAxes();
    FOR_RANGE(size_t, i, 0, shape_dim) {
      KernelUtil<DeviceType::kGPU, int64_t>::Set(ctx, out_blob->shape().At(i), shape_ptr + i);
    }
    const int64_t elem_cnt = num_updates * block_size;
    GpuForward<T, K>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, updates_blob->dptr<T>(), shape_ptr, shape_dim, indices_blob->dptr<K>(),
            indices_blob->shape().dim_vec().back(), block_size, out_blob->mut_dptr<T>());
  }

  static void Backward(DeviceCtx* ctx, const Blob* out_diff_blob, int64_t* shape_ptr,
                       const Blob* indices_blob, const int64_t num_updates,
                       const int64_t block_size, Blob* updates_diff_blob, Blob* in_diff_blob) {
    CHECK_NOTNULL(shape_ptr);
    const int64_t shape_dim = in_diff_blob->shape().NumAxes();
    FOR_RANGE(size_t, i, 0, shape_dim) {
      KernelUtil<DeviceType::kGPU, int64_t>::Set(ctx, in_diff_blob->shape().At(i), shape_ptr + i);
    }
    const int64_t elem_cnt = num_updates * block_size;
    GpuBackward<T, K>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, out_diff_blob->dptr<T>(), shape_ptr, shape_dim, indices_blob->dptr<K>(),
            indices_blob->shape().dim_vec().back(), block_size, updates_diff_blob->mut_dptr<T>(),
            in_diff_blob->mut_dptr<T>());
  }
};

#define MAKE_LOCAL_SCATTER_ND_UPDATE_KERNEL_UTIL_ENTRY(value_type_pair, indices_type_pair) \
  template struct LocalScatterNdUpdateKernelUtil<                                          \
      DeviceType::kGPU, OF_PP_PAIR_FIRST(value_type_pair), OF_PP_PAIR_FIRST(indices_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_LOCAL_SCATTER_ND_UPDATE_KERNEL_UTIL_ENTRY,
                                 FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
