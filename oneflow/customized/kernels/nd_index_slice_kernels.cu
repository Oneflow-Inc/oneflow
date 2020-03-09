#include "oneflow/customized/kernels/nd_index_slice_kernels.h"
#include "oneflow/core/kernel/util/cuda_kernel_util.h"

namespace oneflow {

namespace {

template<typename T, typename I>
__global__ void CudaGatherNd(NdIndexSliceParams<T, I> params) {
  GatherNdFunctor<T, I>::Invoke(params.num_slices * params.slice_size, params.slice_size,
                                params.index_ndims, params.dense_shape, params.indices_dptr,
                                params.dense_dptr, params.slices_dptr);
}

template<typename T, typename I, template<DeviceType, typename> class Opt>
__global__ void CudaScatterNd(NdIndexSliceParams<T, I> params) {
  ScatterNdFunctor<T, I, Opt<DeviceType::kGPU, T>>::Invoke(
      params.num_slices * params.slice_size, params.slice_size, params.index_ndims,
      params.dense_shape, params.indices_dptr, params.slices_dptr, params.dense_dptr);
}

}  // namespace

template<typename T, typename I>
struct GatherNdImpl<DeviceType::kGPU, T, I> {
  static void Apply(DeviceCtx* ctx, NdIndexSliceParams<T, I>* params) {
    RUN_CUDA_KERNEL((CudaGatherNd<T, I>), ctx, params->num_slices * params->slice_size, *params);
  }
};

template<typename T, typename I, template<DeviceType, typename> class Opt>
struct ScatterNdImpl<DeviceType::kGPU, T, I, Opt> {
  static void Apply(DeviceCtx* ctx, NdIndexSliceParams<T, I>* params) {
    RUN_CUDA_KERNEL((CudaScatterNd<T, I, Opt>), ctx, params->num_slices * params->slice_size,
                    *params);
  }
};

template<typename T>
struct ScatterNdReduceAdd<DeviceType::kGPU, T> {
  __device__ __forceinline__ static void Invoke(const T* x, T* y) { gpu_atomic_add(y, *x); }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_GATHER_SCATTER_ND_IMPL, (DeviceType::kGPU),
                                 GATHER_ND_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace oneflow
