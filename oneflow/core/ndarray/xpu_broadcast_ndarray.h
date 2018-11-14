#ifndef ONEFLOW_CORE_NDARRAY_XPU_BROADCAST_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_XPU_BROADCAST_NDARRAY_H_

#include "oneflow/core/ndarray/xpu_var_ndarray.h"

namespace oneflow {

template<typename T, int NDIMS>
struct XpuBroadcastNdarrayUtil;

template<typename T>
class XpuBroadcastNdarray final {
 public:
  OF_DEVICE_FUNC XpuBroadcastNdarray(const ExecShape& shape, const XpuVarNdarray<T>& var)
      : shape_(&shape), var_(&var) {}
  OF_DEVICE_FUNC ~XpuBroadcastNdarray() = default;

  template<int NDIMS>
  OF_DEVICE_FUNC T Get(int64_t offset) const {
    return var_->Get<NDIMS>(XpuBroadcastNdarrayUtil<T, NDIMS>::OriginVarOffset(this, offset));
  }

  OF_DEVICE_FUNC const ExecShape& shape() const { return *shape_; }
  OF_DEVICE_FUNC const XpuVarNdarray<T>& var() const { return *var_; }

 private:
  const ExecShape* shape_;
  const XpuVarNdarray<T>* var_;
};

template<typename T>
struct XpuBroadcastNdarrayUtil<T, 1> final {
  OF_DEVICE_FUNC static int64_t OriginVarOffset(const XpuBroadcastNdarray<T>* ba, int64_t offset) {
    return offset % ba->shape().At(0);
  }
};

template<typename T>
struct XpuBroadcastNdarrayUtil<T, 2> final {
  OF_DEVICE_FUNC static int64_t OriginVarOffset(const XpuBroadcastNdarray<T>* ba, int64_t offset) {
    int64_t dim0 = 0;
    int64_t dim1 = 0;
    ba->shape().Offset2Dims(offset, &dim0, &dim1);
    return ba->var().shape().Dims2Offset(dim0 % ba->var().shape().At(0),
                                         dim1 % ba->var().shape().At(1));
  }
};

template<typename T>
struct XpuBroadcastNdarrayUtil<T, 3> final {
  OF_DEVICE_FUNC static int64_t OriginVarOffset(const XpuBroadcastNdarray<T>* ba, int64_t offset) {
    int64_t dim0 = 0;
    int64_t dim1 = 0;
    int64_t dim2 = 0;
    ba->shape().Offset2Dims(offset, &dim0, &dim1, &dim2);
    return ba->var().shape().Dims2Offset(dim0 % ba->var().shape().At(0),
                                         dim1 % ba->var().shape().At(1),
                                         dim2 % ba->var().shape().At(2));
  }
};

template<typename T>
struct XpuBroadcastNdarrayUtil<T, 4> final {
  OF_DEVICE_FUNC static int64_t OriginVarOffset(const XpuBroadcastNdarray<T>* ba, int64_t offset) {
    int64_t dim0 = 0;
    int64_t dim1 = 0;
    int64_t dim2 = 0;
    int64_t dim3 = 0;
    ba->shape().Offset2Dims(offset, &dim0, &dim1, &dim2, &dim3);
    return ba->var().shape().Dims2Offset(
        dim0 % ba->var().shape().At(0), dim1 % ba->var().shape().At(1),
        dim2 % ba->var().shape().At(2), dim3 % ba->var().shape().At(3));
  }
};

template<typename T>
struct XpuBroadcastNdarrayUtil<T, 5> final {
  OF_DEVICE_FUNC static int64_t OriginVarOffset(const XpuBroadcastNdarray<T>* ba, int64_t offset) {
    int64_t dim0 = 0;
    int64_t dim1 = 0;
    int64_t dim2 = 0;
    int64_t dim3 = 0;
    int64_t dim4 = 0;
    ba->shape().Offset2Dims(offset, &dim0, &dim1, &dim2, &dim3, &dim4);
    return ba->var().shape().Dims2Offset(
        dim0 % ba->var().shape().At(0), dim1 % ba->var().shape().At(1),
        dim2 % ba->var().shape().At(2), dim3 % ba->var().shape().At(3),
        dim4 % ba->var().shape().At(4));
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_BROADCAST_NDARRAY_H_
