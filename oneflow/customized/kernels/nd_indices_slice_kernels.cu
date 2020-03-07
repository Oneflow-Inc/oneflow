#include "oneflow/customized/kernels/nd_indices_slice_util.h"
#include "oneflow/core/kernel/util/cuda_kernel_util.h"

namespace oneflow {

namespace {

template<typename T, typename I>
__global__ void CudaGatherNd(NdIndicesSliceParams<T, I> params) {
  GatherNdFunctor<T, I>::Invoke(params.num_segms * params.segm_size, params.segm_size,
                                params.segm_dim, params.shape, params.indices, params.dense,
                                params.sparse);
}

template<typename T, typename I, template<DeviceType, typename> class Opt>
__global__ void CudaScatterNd(NdIndicesSliceParams<T, I> params) {
  ScatterNdFunctor<T, I, Opt<DeviceType::kGPU, T>>::Invoke(
      params.num_segms * params.segm_size, params.segm_size, params.segm_dim, params.shape,
      params.indices, params.sparse, params.dense);
}

}  // namespace

template<typename T, typename I>
struct GatherNdOnDevice<DeviceType::kGPU, T, I> {
  static void Run(DeviceCtx* ctx, NdIndicesSliceParams<T, I>* params) {
    RUN_CUDA_KERNEL((CudaGatherNd<T, I>), ctx, params->num_segms * params->segm_size, *params);
  }
};

template<typename T, typename I, template<DeviceType, typename> class Opt>
struct ScatterNdOnDevice<DeviceType::kGPU, T, I, Opt> {
  static void Run(DeviceCtx* ctx, NdIndicesSliceParams<T, I>* params) {
    RUN_CUDA_KERNEL((CudaScatterNd<T, I, Opt>), ctx, params->num_segms * params->segm_size,
                    *params);
  }
};

template<typename T>
struct ScatterNdAddOpt<DeviceType::kGPU, T> {
  __device__ __forceinline__ static void Invoke(const T* x, T* y) { gpu_atomic_add(y, *x); }
};

#define INSTANTIATE_GPU_GATHER_SCATTER_ND_UTIL(dtype_pair, itype_pair)                 \
  template struct GatherNdOnDevice<DeviceType::kGPU, OF_PP_PAIR_FIRST(dtype_pair),     \
                                   OF_PP_PAIR_FIRST(itype_pair)>;                      \
  template struct ScatterNdOnDevice<DeviceType::kGPU, OF_PP_PAIR_FIRST(dtype_pair),    \
                                    OF_PP_PAIR_FIRST(itype_pair), ScatterNdUpdateOpt>; \
  template struct ScatterNdOnDevice<DeviceType::kGPU, OF_PP_PAIR_FIRST(dtype_pair),    \
                                    OF_PP_PAIR_FIRST(itype_pair), ScatterNdAddOpt>;

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_GPU_GATHER_SCATTER_ND_UTIL, GATHER_ND_DATA_TYPE_SEQ,
                                 GATHER_ND_INDEX_TYPE_SEQ)
}  // namespace oneflow
