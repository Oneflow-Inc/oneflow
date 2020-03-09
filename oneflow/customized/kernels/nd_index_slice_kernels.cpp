#include "oneflow/customized/kernels/nd_index_slice_kernels.h"

namespace oneflow {

template<typename T, typename I>
struct GatherNdImpl<DeviceType::kCPU, T, I> {
  static void Apply(DeviceCtx* ctx, NdIndexSliceParams<T, I>* params) {
    GatherNdFunctor<T, I>::Invoke(params->num_slices * params->slice_size, params->slice_size,
                                  params->index_ndims, params->dense_shape, params->indices_dptr,
                                  params->dense_dptr, params->slices_dptr);
  }
};

template<typename T, typename I, template<DeviceType, typename> class Opt>
struct ScatterNdImpl<DeviceType::kCPU, T, I, Opt> {
  static void Apply(DeviceCtx* ctx, NdIndexSliceParams<T, I>* params) {
    ScatterNdFunctor<T, I, Opt<DeviceType::kCPU, T>>::Invoke(
        params->num_slices * params->slice_size, params->slice_size, params->index_ndims,
        params->dense_shape, params->indices_dptr, params->slices_dptr, params->dense_dptr);
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
