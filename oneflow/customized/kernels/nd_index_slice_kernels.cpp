#include "oneflow/customized/kernels/nd_index_slice_kernels.h"

namespace oneflow {

template<typename T, typename I>
struct GatherNdFunctor<DeviceType::kCPU, T, I> final {
  void operator()(DeviceCtx* ctx, const NdIndexSliceArgs<T, I>& args, const I* indices,
                  const T* dense, T* slices) const {
    DoGatherNd(args.num_slices * args.slice_size, args.slice_size, args.index_ndims,
               args.dense_shape, indices, dense, slices);
  }
};

template<typename T, typename I>
struct ScatterNdAddFunctor<DeviceType::kCPU, T, I> final {
  void operator()(DeviceCtx* ctx, const NdIndexSliceArgs<T, I>& args, const I* indices,
                  const T* slices, T* dense) const {
    DoScatterNdAdd<DeviceType::kCPU>(args.num_slices * args.slice_size, args.slice_size,
                                     args.index_ndims, args.dense_shape, indices, slices, dense);
  }
};

template<typename T, typename I>
struct ZeroByNdIndexFunctor<DeviceType::kCPU, T, I> final {
  void operator()(DeviceCtx* ctx, const NdIndexSliceArgs<T, I>& args, const I* indices,
                  T* dense) const {
    DoZeroByNdIndex(args.num_slices * args.slice_size, args.slice_size, args.index_ndims,
                    args.dense_shape, indices, dense);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_ND_INDEX_SLICE_FUNCTORS, (DeviceType::kCPU),
                                 ARITHMETIC_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_ND_INDEX_SLICE_KERNELS, (DeviceType::kCPU),
                                 ARITHMETIC_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace oneflow
