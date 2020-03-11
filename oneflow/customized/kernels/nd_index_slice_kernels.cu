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

#define GPU_ATOMIC_ADD_SUPPORTED_DATA_TYPE_SEQ \
  FLOATING_DATA_TYPE_SEQ                       \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_GATHER_ND_IMPL, (DeviceType::kGPU),
                                 ARITHMETIC_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_GATHER_ND_REDUCE_REPLACE_IMPL, (DeviceType::kGPU),
                                 ARITHMETIC_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_GATHER_ND_REDUCE_ADD_IMPL, (DeviceType::kGPU),
                                 GPU_ATOMIC_ADD_SUPPORTED_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_GATHER_ND_KERNELS, (DeviceType::kGPU),
                                 ARITHMETIC_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SCATTER_ND_KERNELS, (DeviceType::kGPU),
                                 GPU_ATOMIC_ADD_SUPPORTED_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_TENSOR_GATHER_ND_UPDATE_KERNELS, (DeviceType::kGPU),
                                 ARITHMETIC_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_TENSOR_GATHER_ND_ADD_KERNELS, (DeviceType::kGPU),
                                 GPU_ATOMIC_ADD_SUPPORTED_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && CUDA_VERSION >= 10000

template<>
struct ScatterNdReduceAdd<DeviceType::kGPU, float16> {
  __device__ __forceinline__ static void Invoke(const float16* x, float16* y) {
    gpu_atomic_add(reinterpret_cast<half*>(y), *(reinterpret_cast<const half*>(x)));
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_GATHER_SCATTER_ND_IMPL, (DeviceType::kGPU),
                                 FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_ND_INDEX_SLICE_KERNELS, (DeviceType::kGPU),
                                 FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

#endif

}  // namespace oneflow
