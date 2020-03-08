#ifndef ONEFLOW_CUSTOMIZED_KERNELS_ND_INDEX_SLICE_UTIL_H_
#define ONEFLOW_CUSTOMIZED_KERNELS_ND_INDEX_SLICE_UTIL_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/xpu_util.h"

namespace oneflow {

template<typename T, typename I>
struct NdIndexSliceParams {
  static const size_t kMaxDims = 8;
  int64_t num_slices;
  int64_t slice_size;
  int64_t index_ndims;
  int64_t dense_shape[kMaxDims];
  T* dense_dptr;
  T* slices_dptr;
  const I* indices_dptr;
};

template<typename T, typename I>
inline NdIndexSliceParams<T, I> ConstructNdIndexSliceParams(user_op::Tensor* dense,
                                                            user_op::Tensor* slices,
                                                            const user_op::Tensor* indices) {
  NdIndexSliceParams<T, I> params;
  std::memset(&params, 0, sizeof(NdIndexSliceParams<T, I>));
  params.num_slices = indices->shape().Count(0, indices->shape().NumAxes() - 1);
  params.index_ndims = indices->shape().At(indices->shape().NumAxes() - 1);
  params.slice_size = slices->shape().Count(indices->shape().NumAxes() - 1);
  FOR_RANGE(int64_t, i, 0, dense->shape().NumAxes()) {
    params.dense_shape[i] = dense->shape().At(i);
  }
  params.dense_dptr = dense->mut_dptr<T>();
  params.slices_dptr = slices->mut_dptr<T>();
  params.indices_dptr = indices->dptr<I>();
  return params;
}

template<DeviceType device_type, typename T, typename I>
struct GatherNdImpl {
  static void Apply(DeviceCtx* ctx, NdIndexSliceParams<T, I>* params);
};

template<DeviceType device_type, typename T, typename I, template<DeviceType, typename> class Opt>
struct ScatterNdImpl {
  static void Apply(DeviceCtx* ctx, NdIndexSliceParams<T, I>* params);
};

// If values in output is to be updated more than once, because there are duplicate entries in
// indices, the order at which the updates happen for each value is undefined.
template<DeviceType device_type, typename T>
struct ScatterNdReduceReplace {
  OF_DEVICE_FUNC static void Invoke(const T* x, T* y) { *y = *x; }
};

template<DeviceType device_type, typename T>
struct ScatterNdReduceAdd {
  OF_DEVICE_FUNC static void Invoke(const T* x, T* y);
};

template<DeviceType device_type, typename T, typename I>
struct NdIndicesSliceUtil final {
  static void GatherNd(DeviceCtx* ctx, NdIndexSliceParams<T, I>* params) {
    GatherNdImpl<device_type, T, I>::Apply(ctx, params);
  }

  static void ScatterNdUpdate(DeviceCtx* ctx, NdIndexSliceParams<T, I>* params) {
    ScatterNdReduce<ScatterNdReduceReplace>(ctx, params);
  }

  static void ScatterNdAdd(DeviceCtx* ctx, NdIndexSliceParams<T, I>* params) {
    ScatterNdReduce<ScatterNdReduceAdd>(ctx, params);
  }

 private:
  template<template<DeviceType, typename> class Opt>
  static void ScatterNdReduce(DeviceCtx* ctx, NdIndexSliceParams<T, I>* params) {
    ScatterNdImpl<device_type, T, I, Opt>::Apply(ctx, params);
  }
};

template<typename I>
struct SliceOffsetInDenseWithNdIndex {
  OF_DEVICE_FUNC static int64_t Compute(int64_t batch_size, int64_t index_ndims,
                                        const int64_t* dense_shape, const I* indices, int64_t n) {
    int64_t batch_idx = n / batch_size;
    const I* cur_nd_index_ptr = indices + batch_idx * index_ndims;
    int64_t offset = 0;
    int64_t product = 1;
    for (int64_t i = index_ndims - 1; i >= 0; --i) {
      offset += cur_nd_index_ptr[i] * product;
      product *= dense_shape[i];
    }
    return offset * batch_size + n % batch_size;
  }
};

template<typename T, typename I>
struct GatherNdFunctor {
  OF_DEVICE_FUNC static void Invoke(int64_t elem_cnt, int64_t slice_size, int64_t index_ndims,
                                    const int64_t* dense_shape, const I* indices, const T* dense,
                                    T* slices) {
    XPU_1D_KERNEL_LOOP(i, elem_cnt) {
      int64_t offset = SliceOffsetInDenseWithNdIndex<I>::Compute(slice_size, index_ndims,
                                                                 dense_shape, indices, i);
      slices[i] = dense[offset];
    }
  }
};

template<typename T, typename I, typename Opt>
struct ScatterNdFunctor;

template<DeviceType device_type, typename T, typename I, template<DeviceType, typename> class Opt>
struct ScatterNdFunctor<T, I, Opt<device_type, T>> {
  OF_DEVICE_FUNC static void Invoke(int64_t elem_cnt, int64_t slice_size, int64_t index_ndims,
                                    const int64_t* dense_shape, const I* indices, const T* slices,
                                    T* dense) {
    XPU_1D_KERNEL_LOOP(i, elem_cnt) {
      int64_t offset = SliceOffsetInDenseWithNdIndex<I>::Compute(slice_size, index_ndims,
                                                                 dense_shape, indices, i);
      Opt<device_type, T>::Invoke(slices + i, dense + offset);
    }
  }
};

#define INSTANTIATE_GATHER_SCATTER_ND_IMPL(device_type_v, dtype_pair, itype_pair)      \
  template struct GatherNdImpl<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),            \
                               OF_PP_PAIR_FIRST(itype_pair)>;                          \
  template struct ScatterNdImpl<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),           \
                                OF_PP_PAIR_FIRST(itype_pair), ScatterNdReduceReplace>; \
  template struct ScatterNdImpl<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),           \
                                OF_PP_PAIR_FIRST(itype_pair), ScatterNdReduceAdd>;

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_ND_INDEX_SLICE_UTIL_H_
