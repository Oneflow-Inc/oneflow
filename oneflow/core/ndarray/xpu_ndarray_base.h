#ifndef ONEFLOW_CORE_NDARRAY_XPU_NDARRAY_BASE_H_
#define ONEFLOW_CORE_NDARRAY_XPU_NDARRAY_BASE_H_

namespace oneflow {

template<typename T, const T (*unary_func)(const T), typename X>
class XpuUnaryFuncNdarray;
template<typename T, const T (*binary_func)(const T, const T), typename A, typename B>
class XpuBinaryFuncNdarray;
template<typename T>
class XpuBroadcastNdarray;
template<typename T, int, typename X>
class XpuTransposeNdarray;
template<typename T, int, typename X>
class XpuReshapeNdarray;

template<typename DerivedT, typename T>
class XpuNdarrayBase {
 public:
  OF_DEVICE_FUNC XpuNdarrayBase() = default;
  OF_DEVICE_FUNC ~XpuNdarrayBase() = default;

  template<const T (*unary_func)(const T)>
  OF_DEVICE_FUNC XpuUnaryFuncNdarray<T, unary_func, DerivedT> UnaryFunc() const {
    return XpuUnaryFuncNdarray<T, unary_func, DerivedT>(*static_cast<const DerivedT*>(this));
  }
  template<const T (*binary_func)(const T, const T), typename X>
  OF_DEVICE_FUNC XpuBinaryFuncNdarray<T, binary_func, DerivedT, X> BinaryFunc(const X& x) const {
    return XpuBinaryFuncNdarray<T, binary_func, DerivedT, X>(*static_cast<const DerivedT*>(this),
                                                             x);
  }
  OF_DEVICE_FUNC XpuBroadcastNdarray<const T> Broadcast(const XpuShape& shape) const {
    return XpuBroadcastNdarray<const T>(shape, *static_cast<const DerivedT*>(this));
  }
  template<int NDIMS>
  OF_DEVICE_FUNC XpuTransposeNdarray<T, NDIMS, DerivedT> Transpose(
      const int64_t perm[NDIMS]) const {
    return XpuTransposeNdarray<T, NDIMS, DerivedT>(*static_cast<const DerivedT*>(this), perm);
  }
  template<int NDIMS>
  OF_DEVICE_FUNC XpuReshapeNdarray<T, NDIMS, DerivedT> Reshape(const int64_t shape[NDIMS]) {
    return XpuReshapeNdarray<T, NDIMS, DerivedT>(*static_cast<const DerivedT*>(this), shape);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_NDARRAY_BASE_H_
