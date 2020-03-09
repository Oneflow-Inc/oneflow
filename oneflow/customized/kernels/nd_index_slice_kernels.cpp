#include "oneflow/customized/kernels/nd_index_slice_kernels.h"

namespace oneflow {

template<typename T, typename I>
struct GatherNdImpl<DeviceType::kCPU, T, I> {
  static void Apply(DeviceCtx* ctx, const NdIndexSliceArgs<T, I>& args, const I* indices,
                    const T* dense, T* slices) {
    GatherNdFunctor<T, I>::Invoke(args.num_slices * args.slice_size, args.slice_size,
                                  args.index_ndims, args.dense_shape, indices, dense, slices);
  }
};

template<typename T, typename I, template<DeviceType, typename> class Opt>
struct ScatterNdImpl<DeviceType::kCPU, T, I, Opt> {
  static void Apply(DeviceCtx* ctx, const NdIndexSliceArgs<T, I>& args, const I* indices,
                    const T* slices, T* dense) {
    ScatterNdFunctor<T, I, Opt<DeviceType::kCPU, T>>::Invoke(
        args.num_slices * args.slice_size, args.slice_size, args.index_ndims, args.dense_shape,
        indices, slices, dense);
  }
};

template<typename T>
struct ScatterNdReduceAdd<DeviceType::kCPU, T> {
  OF_DEVICE_FUNC static void Invoke(const T* x, T* y) { *y += *x; }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_GATHER_SCATTER_ND_IMPL, (DeviceType::kCPU),
                                 GATHER_ND_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_ND_INDEX_SLICE_KERNELS, DEVICE_TYPE_SEQ,
                                 GATHER_ND_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace oneflow
