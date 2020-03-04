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
  params.indices = sparse->dptr<I>();
}

template<DeviceType device_type, typename T, typename I>
struct GatherNdOnDevice;

template<DeviceType device_type, typename T, typename I, template<typename> class func>
struct ScatterNdOnDevice;

// template<typename T>
// struct ApplyUpdate {
//   OF_DEVICE_FUNC static void Invoke(const T* in, T* out) { *out = *in; }
// };

template<DeviceType device_type, typename T, typename I>
struct NdIndicesSliceUtil final {
  static void GatherNd(DeviceCtx* ctx, NdIndicesSliceParams<T, I>* params) {
    GatherNdOnDevice<device_type, T, I>::Run(ctx, params);
  }

  // static void ScatterNdUpdate(DeviceCtx* ctx, const Blob* indices, const Blob* sparse,
  //                             const int64_t* dense_shape, Blob* dense) {
  //   ScatterNdApply<ApplyUpdate>(ctx, indices, sparse, dense_shape, dense);
  // }

 private:
  template<template<typename> class func>
  static void ScatterNdApply(DeviceCtx* ctx, const Blob* indices, const Blob* sparse,
                             const int64_t* dense_shape, Blob* dense) {
    int64_t num_segms = indices->shape().Count(0, indices->shape().NumAxes() - 1);
    int64_t segm_size = sparse->shape().Count(indices->shape().NumAxes() - 1);
    int64_t segm_dim = indices->shape().At(indices->shape().NumAxes() - 1);
    ScatterNdOnDevice<device_type, T, I, func>::Run(ctx, num_segms, segm_size, segm_dim,
                                                    indices->dptr<I>(), dense_shape,
                                                    sparse->dptr<T>(), dense->mut_dptr<T>());
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

template<typename T, typename I, template<typename> class func>
struct ScatterNdFunctor {
  OF_DEVICE_FUNC static void Invoke(int64_t elem_cnt, int64_t segm_size, int64_t segm_dim,
                                    const I* indices, const int64_t* shape, const T* sparse,
                                    T* dense) {
    XPU_1D_KERNEL_LOOP(idx, elem_cnt) {
      int64_t offset = IndicesOffset<I>::Compute(segm_size, segm_dim, shape, indices, idx);
      func<T>::Invoke(sparse + idx, dense + offset);
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_ND_INDICES_SLICE_UTIL_H_
