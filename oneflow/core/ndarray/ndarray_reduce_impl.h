#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_IMPL_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_IMPL_H_

#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace ndarray {

template<DeviceType device_type, typename T>
struct MatrixRowReduce final {
  static bool Matched(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) { return false; }
  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    UNIMPLEMENTED();
  }
};

template<DeviceType device_type, typename T>
struct MatrixColReduce final {
  static bool Matched(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) { return false; }
  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    UNIMPLEMENTED();
  }
};

template<typename T>
struct MatrixRowReduce<DeviceType::kGPU, T> final {
  static bool Matched(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) {
    const auto& x_squeezed = SqueezeRight(x.shape());
    const auto& y_squeezed = SqueezeRight(y.shape());
    if (x_squeezed.NumAxes() == 0) { return false; }
    for (int i = 0; i < x_squeezed.NumAxes(); ++i) {
      if (x_squeezed.At(i) != y_squeezed.At(i)) { return false; }
    }
    CHECK_EQ(y.shape().ElemNum() % x.shape().ElemNum(), 0);
    return true;
  }

  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    CHECK(Matched(y, x));
    int64_t num_rows = x.shape().ElemNum();
    int64_t num_cols = y.shape().ElemNum() / x.shape().ElemNum();
  }

 private:
  static XpuShape SqueezeRight(const XpuShape& shape) {
    TODO();
    return shape;
  }
};

template<typename T>
struct MatrixColReduce<DeviceType::kGPU, T> final {
  static bool Matched(const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x) {
    const auto& x_squeezed = SqueezeLeft(x.shape());
    const auto& y_squeezed = SqueezeLeft(y.shape());
    if (x_squeezed.NumAxes() == 0) { return false; }
    for (int i = 0; i < x_squeezed.NumAxes(); ++i) {
      if (x_squeezed.At(x_squeezed.NumAxes() - 1 - i)
          != y_squeezed.At(y_squeezed.NumAxes() - 1 - i)) {
        return false;
      }
    }
    return true;
  }

  static void Reduce(DeviceCtx* ctx, const XpuVarNdarray<T>& y, const XpuVarNdarray<const T>& x,
                     const XpuVarNdarray<T>& tmp_storage) {
    CHECK(Matched(y, x));
  }

 private:
  static XpuShape SqueezeLeft(const XpuShape& shape) {
    TODO();
    return shape;
  }
};

}  // namespace ndarray

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_REDUCE_IMPL_H_
