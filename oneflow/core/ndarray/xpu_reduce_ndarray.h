#ifndef ONEFLOW_CORE_NDARRAY_XPU_REDUCE_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_XPU_REDUCE_NDARRAY_H_

#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/unary_func.h"

namespace oneflow {

template<typename T, int NDIMS, typename X>
class XpuReduceNdarray final {
 public:
  OF_DEVICE_FUNC XpuReduceNdarray(const ExecShape& shape, const X& x, XpuVarNdarray<T>* storage)
      : shape_(shape), data_(XpuVarNdarray<T>(x.shape(), storage->mut_ptr())) {
    data_.template Assign<NDIMS>(x);
    Reduce();
  }

  template<int ndims, typename = typename std::enable_if<ndims == NDIMS>::type>
  OF_DEVICE_FUNC int64_t Get(int64_t offset) const {
    int64_t dim[NDIMS];
    ExecShapeUtil<NDIMS>::Offset2DimVec(shape_, offset, dim);
    return data_.template Get<NDIMS>(ExecShapeUtil<NDIMS>::DimVec2Offset(data_.shape(), dim));
  }

 private:
  OF_DEVICE_FUNC void Reduce() {
    ExecShape cur_shape(data_.shape());
    for (int i = 0; i < NDIMS; ++i) {
      if (shape_.At(i) == data_.shape().At(i)) { continue; }
      ReduceAxisToPowerOf2(i, &cur_shape);
      ReduceAxisOnceHalf(i, &cur_shape);
    }
  }

  OF_DEVICE_FUNC void ReduceAxisOnceHalf(int axis, ExecShape* cur_shape) {
    while (cur_shape->At(axis) > 1) {
      int64_t new_dim_val = cur_shape->At(axis) / 2;
      cur_shape->Set(axis, new_dim_val);
      XPU_1D_KERNEL_LOOP(i, cur_shape->ElemNum()) {
        int64_t dim[NDIMS];
        ExecShapeUtil<NDIMS>::Offset2DimVec(*cur_shape, i, dim);
        int64_t small_offset = ExecShapeUtil<NDIMS>::DimVec2Offset(data_.shape(), dim);
        T* small_offset_ptr = data_.template Mut<NDIMS>(small_offset);
        dim[axis] += new_dim_val;
        int64_t big_offset = ExecShapeUtil<NDIMS>::DimVec2Offset(data_.shape(), dim);
        const T big_offset_val = data_.template Get<NDIMS>(big_offset);
        *small_offset_ptr += big_offset_val;
      }
      XpuSyncThreads();
    }
  }

  OF_DEVICE_FUNC void ReduceAxisToPowerOf2(int axis, ExecShape* cur_shape) {
    int64_t shift = UnaryFuncLog2(cur_shape->At(axis));
    int64_t new_dim_val = UnaryFuncExp2(shift);
    cur_shape->Set(axis, new_dim_val);
    XPU_1D_KERNEL_LOOP(i, cur_shape->ElemNum()) {
      int64_t dim[NDIMS];
      ExecShapeUtil<NDIMS>::Offset2DimVec(*cur_shape, i, dim);
      if (dim[axis] + new_dim_val < data_.shape().At(axis)) {
        int64_t small_offset = ExecShapeUtil<NDIMS>::DimVec2Offset(data_.shape(), dim);
        T* small_offset_ptr = data_.template Mut<NDIMS>(small_offset);
        dim[axis] += new_dim_val;
        int64_t big_offset = ExecShapeUtil<NDIMS>::DimVec2Offset(data_.shape(), dim);
        const T big_offset_val = data_.template Get<NDIMS>(big_offset);
        *small_offset_ptr += big_offset_val;
      }
    }
    XpuSyncThreads();
  }

  ExecShape shape_;
  XpuVarNdarray<T> data_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_REDUCE_NDARRAY_H_
