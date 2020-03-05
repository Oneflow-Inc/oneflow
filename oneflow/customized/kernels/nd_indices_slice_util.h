#ifndef ONEFLOW_CUSTOMIZED_KERNELS_ND_INDICES_SLICE_UTIL_H_
#define ONEFLOW_CUSTOMIZED_KERNELS_ND_INDICES_SLICE_UTIL_H_

// #include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/framework/framework.h"

// #define SCATTER_ND_FUNC_NAME_SEQ (Update)(Add)

namespace oneflow {

template<typename T, typename I>
struct NdIndicesSliceParams {
  static const size_t kMaxDims = 8;
  int64_t num_segms;
  int64_t segm_size;
  int64_t segm_dim;
  int64_t shape[kMaxDims];
  T* dense;
  T* sparse;
  const I* indices;
};

template<typename T, typename I>
inline NdIndicesSliceParams<T, I> ConstructNdIndicesSliceParams(user_op::Tensor* dense,
                                                                user_op::Tensor* sparse,
                                                                const user_op::Tensor* indices) {
  NdIndicesSliceParams<T, I> params;
  std::memset(&params, 0, sizeof(NdIndicesSliceParams<T, I>));
  params.num_segms = indices->shape().Count(0, indices->shape().NumAxes() - 1);
  params.segm_size = sparse->shape().Count(indices->shape().NumAxes() - 1);
  params.segm_dim = indices->shape().At(indices->shape().NumAxes() - 1);
  FOR_RANGE(int64_t, i, 0, dense->shape().NumAxes()) { params.shape[i] = dense->shape().At(i); }
  params.dense = dense->mut_dptr<T>();
  params.sparse = sparse->mut_dptr<T>();
  params.indices = indices->dptr<I>();
  return params;
}

template<DeviceType device_type, typename T, typename I>
struct GatherNdOnDevice;

template<DeviceType device_type, typename T, typename I, template<DeviceType, typename> class Opt>
struct ScatterNdOnDevice;

// If values in output is to be updated more than once, because there are duplicate entries in
// indices, the order at which the updates happen for each value is undefined.
template<DeviceType device_type, typename T>
struct ScatterNdUpdateOpt {
  OF_DEVICE_FUNC static void Invoke(const T* x, T* y) { *y = *x; }
};

template<DeviceType device_type, typename T>
struct ScatterNdAddOpt;

template<DeviceType device_type, typename T, typename I>
struct NdIndicesSliceUtil final {
  static void GatherNd(DeviceCtx* ctx, NdIndicesSliceParams<T, I>* params) {
    GatherNdOnDevice<device_type, T, I>::Run(ctx, params);
  }

  static void ScatterNdUpdate(DeviceCtx* ctx, NdIndicesSliceParams<T, I>* params) {
    ScatterNdApplyOpt<ScatterNdUpdateOpt>(ctx, params);
  }

  static void ScatterNdAdd(DeviceCtx* ctx, NdIndicesSliceParams<T, I>* params) {
    ScatterNdApplyOpt<ScatterNdAddOpt>(ctx, params);
  }

 private:
  template<template<DeviceType, typename> class Opt>
  static void ScatterNdApplyOpt(DeviceCtx* ctx, NdIndicesSliceParams<T, I>* params) {
    ScatterNdOnDevice<device_type, T, I, Opt>::Run(ctx, params);
  }
};

template<typename I>
struct IndicesOffset {
  OF_DEVICE_FUNC static int64_t Compute(int64_t segm_size, int64_t segm_dim, const int64_t* shape,
                                        const I* indices, int64_t n) {
    int64_t offset = 0;
    const auto* cur_ids_ptr = indices + (n / segm_size) * segm_dim;
    FOR_RANGE(int64_t, i, 0, segm_dim) {
      assert(cur_ids_ptr[i] < shape[i]);
      int64_t stride = segm_size;
      FOR_RANGE(int64_t, j, i + 1, segm_dim) { stride *= shape[j]; }
      offset += cur_ids_ptr[i] * stride;
    }
    return offset + n % segm_size;
  }
};

template<typename T, typename I>
struct GatherNdFunctor {
  OF_DEVICE_FUNC static void Invoke(int64_t elem_cnt, int64_t segm_size, int64_t segm_dim,
                                    const int64_t* shape, const I* indices, const T* dense,
                                    T* sparse) {
    XPU_1D_KERNEL_LOOP(i, elem_cnt) {
      int64_t offset = IndicesOffset<I>::Compute(segm_size, segm_dim, shape, indices, i);
      sparse[i] = dense[offset];
    }
  }
};

template<typename T, typename I, typename Opt>
struct ScatterNdFunctor;

template<DeviceType device_type, typename T, typename I, template<DeviceType, typename> class Opt>
struct ScatterNdFunctor<T, I, Opt<device_type, T>> {
  OF_DEVICE_FUNC static void Invoke(int64_t elem_cnt, int64_t segm_size, int64_t segm_dim,
                                    const int64_t* shape, const I* indices, const T* sparse,
                                    T* dense) {
    XPU_1D_KERNEL_LOOP(i, elem_cnt) {
      int64_t offset = IndicesOffset<I>::Compute(segm_size, segm_dim, shape, indices, i);
      Opt<device_type, T>::Invoke(sparse + i, dense + offset);
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_ND_INDICES_SLICE_UTIL_H_
