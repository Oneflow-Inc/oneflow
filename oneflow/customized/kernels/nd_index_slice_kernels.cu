#include "oneflow/customized/kernels/nd_index_slice_kernels.h"
#include "oneflow/core/kernel/util/cuda_kernel_util.h"

namespace oneflow {

namespace {

template<typename T, typename I>
__global__ void CudaGatherNd(NdIndexSliceArgs<T, I> args, const I* indices, const T* dense,
                             T* slices) {
  GatherNdFunctor<T, I>::Invoke(args.num_slices * args.slice_size, args.slice_size,
                                args.index_ndims, args.dense_shape, indices, dense, slices);
}

template<typename T, typename I, template<DeviceType, typename> class Opt>
__global__ void CudaScatterNd(NdIndexSliceArgs<T, I> args, const I* indices, const T* slices,
                              T* dense) {
  ScatterNdFunctor<T, I, Opt<DeviceType::kGPU, T>>::Invoke(
      args.num_slices * args.slice_size, args.slice_size, args.index_ndims, args.dense_shape,
      indices, slices, dense);
}

}  // namespace

template<typename T, typename I>
struct GatherNdImpl<DeviceType::kGPU, T, I> {
  static void Apply(DeviceCtx* ctx, const NdIndexSliceArgs<T, I>& args, const I* indices,
                    const T* dense, T* slices) {
    RUN_CUDA_KERNEL((CudaGatherNd<T, I>), ctx, args.num_slices * args.slice_size, args, indices,
                    dense, slices);
  }
};

template<typename T, typename I, template<DeviceType, typename> class Opt>
struct ScatterNdImpl<DeviceType::kGPU, T, I, Opt> {
  static void Apply(DeviceCtx* ctx, const NdIndexSliceArgs<T, I>& args, const I* indices,
                    const T* slices, T* dense) {
    RUN_CUDA_KERNEL((CudaScatterNd<T, I, Opt>), ctx, args.num_slices * args.slice_size, args,
                    indices, slices, dense);
  }
};

template<typename T>
struct ScatterNdReduceAdd<DeviceType::kGPU, T> {
  __device__ __forceinline__ static void Invoke(const T* x, T* y) { gpu_atomic_add(y, *x); }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_GATHER_SCATTER_ND_IMPL, (DeviceType::kGPU),
                                 GATHER_ND_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace oneflow
