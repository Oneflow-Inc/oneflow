#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_CORE_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_CORE_H_

#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/xpu_reduced_ndarray.h"
#include "oneflow/core/ndarray/xpu_transpose_ndarray.h"
#include "oneflow/core/ndarray/xpu_reshape_ndarray.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_ndarray_builder.h"

namespace oneflow {

template<DeviceType device_type, typename T, int NDIMS>
struct NdArrayReduceCoreWrapper final {
  static void ReduceAxis(DeviceCtx* ctx, const XpuReducedNdarray<T, NDIMS>& dst_reduced,
                         const XpuVarNdarray<const T>& x, int axis);
  static void ReduceAxis(DeviceCtx* ctx, const XpuReducedNdarray<T, NDIMS>& dst_reduced,
                         const XpuReducedNdarray<T, NDIMS>& x, int axis);
};

template<typename T, int NDIMS>
struct NdArrayReduceCore final {
  template<typename X>
  OF_DEVICE_FUNC static void ReduceAxis(const XpuReducedNdarray<T, NDIMS>& dst_reduced, const X& x,
                                        int axis) {
    size_t n = dst_reduced.shape().ElemNum();
    int64_t dst_dim_val = dst_reduced.shape().At(axis);
    XPU_1D_KERNEL_LOOP(i, n) {
      T* dst_reduced_ptr = dst_reduced.template Mut(i);
      int64_t coord[NDIMS];
      ExecShapeUtil<NDIMS>::Offset2DimVec(dst_reduced.shape(), i, coord);
      T sum = 0;
      while (coord[axis] < x.shape().At(axis)) {
        sum += x.template Get<NDIMS>(coord);
        coord[axis] += dst_dim_val;
      }
      *dst_reduced_ptr = sum;
    }
  }
  OF_DEVICE_FUNC static void BlockThread2DImplaceReduceSum(const XpuVarNdarray<T>& implace,
                                                           const XpuVarNdarray<const T>& x,
                                                           const int64_t perm[NDIMS],
                                                           int64_t reshape_dim[2]) {
    XpuNdArrayBuilder<T, NDIMS> ndarray;
    const auto& x_transposed = ndarray.Transposed(implace, perm);
    const auto& x_reshaped = ndarray.Reshape<2>(x_transposed, reshape_dim);

    const auto& transposed = ndarray.Transposed(implace, perm);
    const auto& reshaped = ndarray.Reshape<2>(transposed, reshape_dim);

    XPU_BLOAD_THREAD_2D_KERNEL_LOOP(i, j, reshape_dim[0], reshape_dim[1]) {
      int64_t coord[2] = {i, j};
      *(reshaped.template Mut<2>(coord)) = x_reshaped.template Get<2>(coord);
    }
    XpuSyncThreads();
    while (reshape_dim[1] > 1) {
      int64_t old_col_num = reshape_dim[1];
      int64_t new_col_num = old_col_num / 2;
      XPU_BLOAD_THREAD_2D_KERNEL_LOOP(i, j, reshape_dim[0], old_col_num) {
        int64_t coord[2] = {i, j};
        T* ptr = reshaped.template Mut<2>(coord);
        coord[1] += old_col_num;
        if (coord[1] < new_col_num) { *ptr += reshaped.template Get<2>(coord); }
      }
      reshape_dim[1] = new_col_num;
      XpuSyncThreads();
    }
  }
  OF_DEVICE_FUNC static void BlockThread2DReduceSumAndOutput(
      const XpuVarNdarray<T>& y, const XpuVarNdarray<T>& implace_reduced, const int64_t perm[NDIMS],
      int64_t reshape_dim[2]) {
    XpuNdArrayBuilder<T, NDIMS> ndarray;
    const auto& transposed = ndarray.Transposed(implace_reduced, perm);
    const auto& reshaped = ndarray.Reshape<2>(transposed, reshape_dim);
    reshape_dim[1] = 1;
    const auto& out_reshaped = ndarray.Reduced<2>(ExecShape(reshape_dim, 2), reshaped);
    int64_t out_reshape_dim[2] = {y.shape().ElemNum(), reshape_dim[0] / y.shape().ElemNum()};
    while (out_reshape_dim[1] > 1) {
      int64_t old_col_num = out_reshape_dim[1];
      int64_t new_col_num = old_col_num / 2;
      XPU_BLOAD_THREAD_2D_KERNEL_LOOP(i, j, out_reshape_dim[0], old_col_num) {
        int64_t coord[2] = {i, j};
        T* ptr = out_reshaped.template Mut<2>(coord);
        coord[1] += old_col_num;
        if (coord[1] < new_col_num) { *ptr += out_reshaped.template Get<2>(coord); }
      }
      out_reshape_dim[1] = new_col_num;
      XpuSyncThreads();
    }
    XPU_BLOAD_THREAD_2D_KERNEL_LOOP(i, j, out_reshape_dim[0], 1) {
      int64_t coord[2] = {i, 1};
      *(y.template Mut<1>(i)) = out_reshaped.template Get<2>(coord);
    }
  }
  OF_DEVICE_FUNC static void OutputReducedSum(const XpuVarNdarray<T>& y,
                                              const XpuVarNdarray<T>& implace_reduced,
                                              const int64_t perm[NDIMS], int64_t reshape_dim[2]) {
    XpuNdArrayBuilder<T, NDIMS> ndarray;
    const auto& transposed = ndarray.Transposed(implace_reduced, perm);
    const auto& reshaped = ndarray.Reshape<2>(transposed, reshape_dim);
    XPU_1D_KERNEL_LOOP(i, reshaped[0]) {
      int64_t coord[2] = {i, 1};
      *(y.template Mut<1>(i)) = reshaped.template Get<2>(coord);
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_CORE_H_
