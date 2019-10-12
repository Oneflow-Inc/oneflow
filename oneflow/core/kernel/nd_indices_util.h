#ifndef ONEFLOW_CORE_KERNEL_ND_INDICES_UTIL_H_
#define ONEFLOW_CORE_KERNEL_ND_INDICES_UTIL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/ndarray/xpu_util.h"

#define SCATTER_ND_FUNC_NAME_SEQ (Update)(Add)

namespace oneflow {

template<DeviceType device_type, typename T, typename I>
struct GatherNdOnDevice;

template<DeviceType device_type, typename T, typename I, template<typename> class func>
struct ScatterNdOnDevice;

template<typename T>
struct ApplyUpdate {
  OF_DEVICE_FUNC static void Invoke(const T* in, T* out) { *out = *in; }
};

template<DeviceType device_type, typename T, typename I>
struct NdIndicesUtil final {
  /*
  #define DEFINE_SCATTER_ND_FUNC(func_name)                                                   \
    static void ScatterNd##func_name(DeviceCtx* ctx, const Blob* indices, const Blob* sparse, \
                                     const int64_t* dense_shape, Blob* dense) {               \
      ScatterNdApply<Apply##func_name>(ctx, indices, sparse, dense_shape, dense);  \
    }
    OF_PP_FOR_EACH_ATOMIC(DEFINE_SCATTER_ND_FUNC, SCATTER_ND_FUNC_NAME_SEQ)
  #undef DEFINE_SCATTER_ND_FUNC
  */
  static void ScatterNdUpdate(DeviceCtx* ctx, const Blob* indices, const Blob* sparse,
                              const int64_t* dense_shape, Blob* dense) {
    ScatterNdApply<ApplyUpdate>(ctx, indices, sparse, dense_shape, dense);
  }

  static void GatherNd(DeviceCtx* ctx, const Blob* indices, const Blob* dense,
                       const int64_t* dense_shape, Blob* sparse) {
    int64_t num_segms = indices->shape().Count(0, indices->shape().NumAxes() - 1);
    int64_t segm_size = sparse->shape().Count(indices->shape().NumAxes() - 1);
    int64_t segm_dims = indices->shape().At(indices->shape().NumAxes() - 1);
    GatherNdOnDevice<device_type, T, I>::Run(ctx, num_segms, segm_size, segm_dims,
                                             indices->dptr<I>(), dense_shape, dense->dptr<T>(),
                                             sparse->mut_dptr<T>());
  }

 private:
  template<template<typename> class func>
  static void ScatterNdApply(DeviceCtx* ctx, const Blob* indices, const Blob* sparse,
                             const int64_t* dense_shape, Blob* dense) {
    int64_t num_segms = indices->shape().Count(0, indices->shape().NumAxes() - 1);
    int64_t segm_size = sparse->shape().Count(indices->shape().NumAxes() - 1);
    int64_t segm_dims = indices->shape().At(indices->shape().NumAxes() - 1);
    ScatterNdOnDevice<device_type, T, I, func>::Run(ctx, num_segms, segm_size, segm_dims,
                                                    indices->dptr<I>(), dense_shape,
                                                    sparse->dptr<T>(), dense->mut_dptr<T>());
  }
};

template<typename I>
struct IndicesOffset {
  OF_DEVICE_FUNC static int64_t Compute(int64_t segms_size, int64_t segm_dims, const int64_t* shape,
                                        const I* indices, int64_t n) {
    int64_t offset = 0;
    const auto* cur_ids_ptr = indices + (n / segms_size) * segm_dims;
    FOR_RANGE(int64_t, i, 0, segm_dims) {
      assert(cur_ids_ptr[i] < shape[i]);
      int64_t stride = segms_size;
      FOR_RANGE(int64_t, j, i + 1, segm_dims) { stride *= shape[j]; }
      offset += cur_ids_ptr[i] * stride;
    }
    return offset + n % segms_size;
  }
};

template<typename T, typename I, template<typename> class func>
struct ScatterNdFunctor {
  OF_DEVICE_FUNC static void Invoke(int64_t elem_cnt, int64_t segms_size, int64_t segm_dims,
                                    const I* indices, const int64_t* shape, const T* sparse,
                                    T* dense) {
    XPU_1D_KERNEL_LOOP(idx, elem_cnt) {
      int64_t offset = IndicesOffset<I>::Compute(segms_size, segm_dims, shape, indices, idx);
      func<T>::Invoke(sparse + idx, dense + offset);
    }
  }
};

template<typename T, typename I>
struct GatherNdFunctor {
  OF_DEVICE_FUNC static void Invoke(int64_t elem_cnt, int64_t segms_size, int64_t segm_dims,
                                    const I* indices, const int64_t* shape, const T* dense,
                                    T* sparse) {
    XPU_1D_KERNEL_LOOP(i, elem_cnt) {
      int64_t offset = IndicesOffset<I>::Compute(segms_size, segm_dims, shape, indices, i);
      sparse[i] = dense[offset];
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ND_INDICES_UTIL_H_
