#ifndef ONEFLOW_CUSTOMIZED_KERNELS_ND_INDEX_SLICE_UTIL_H_
#define ONEFLOW_CUSTOMIZED_KERNELS_ND_INDEX_SLICE_UTIL_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/xpu_util.h"

namespace oneflow {

template<typename T, typename I>
struct NdIndexSliceArgs {
  static const size_t kMaxDims = 8;
  int64_t num_slices;
  int64_t slice_size;
  int64_t index_ndims;
  int64_t dense_shape[kMaxDims];
};

template<typename T, typename I>
inline NdIndexSliceArgs<T, I> ConstructNdIndexSliceArgs(const user_op::Tensor& dense,
                                                        const user_op::Tensor& slices,
                                                        const user_op::Tensor& indices) {
  NdIndexSliceArgs<T, I> args;
  std::memset(&args, 0, sizeof(NdIndexSliceArgs<T, I>));
  args.num_slices = indices.shape().Count(0, indices.shape().NumAxes() - 1);
  args.index_ndims = indices.shape().At(indices.shape().NumAxes() - 1);
  args.slice_size = slices.shape().Count(indices.shape().NumAxes() - 1);
  FOR_RANGE(int64_t, i, 0, dense.shape().NumAxes()) { args.dense_shape[i] = dense.shape().At(i); }
  return args;
}

template<DeviceType device_type, typename T, typename I>
struct GatherNdImpl {
  static void Apply(DeviceCtx* ctx, const NdIndexSliceArgs<T, I>& args, const I* indices,
                    const T* dense, T* slices);
};

template<DeviceType device_type, typename T, typename I, template<DeviceType, typename> class Opt>
struct ScatterNdImpl {
  static void Apply(DeviceCtx* ctx, const NdIndexSliceArgs<T, I>& args, const I* indices,
                    const T* slices, T* dense);
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
  static void GatherNd(DeviceCtx* ctx, const NdIndexSliceArgs<T, I>& args, const I* indices,
                       const T* dense, T* slices) {
    GatherNdImpl<device_type, T, I>::Apply(ctx, args, indices, dense, slices);
  }

  static void ScatterNdUpdate(DeviceCtx* ctx, const NdIndexSliceArgs<T, I>& args, const I* indices,
                              const T* slices, T* dense) {
    ScatterNdReduce<ScatterNdReduceReplace>(ctx, args, indices, slices, dense);
  }

  static void ScatterNdAdd(DeviceCtx* ctx, const NdIndexSliceArgs<T, I>& args, const I* indices,
                           const T* slices, T* dense) {
    ScatterNdReduce<ScatterNdReduceAdd>(ctx, args, indices, slices, dense);
  }

 private:
  template<template<DeviceType, typename> class Opt>
  static void ScatterNdReduce(DeviceCtx* ctx, const NdIndexSliceArgs<T, I>& args, const I* indices,
                              const T* slices, T* dense) {
    ScatterNdImpl<device_type, T, I, Opt>::Apply(ctx, args, indices, slices, dense);
  }
};

template<typename I>
struct SliceOffsetInDenseWithNdIndex {
  OF_DEVICE_FUNC static int64_t Compute(int64_t slice_size, int64_t index_ndims,
                                        const int64_t* dense_shape, const I* indices, int64_t n) {
    int64_t slice_idx = n / slice_size;
    const I* cur_nd_index_ptr = indices + slice_idx * index_ndims;
    int64_t offset = 0;
    int64_t product = 1;
    for (int64_t i = index_ndims - 1; i >= 0; --i) {
      offset += cur_nd_index_ptr[i] * product;
      product *= dense_shape[i];
    }
    return offset * slice_size + n % slice_size;
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
