#ifndef ONEFLOW_CORE_UNARY_FUNC_NDARRAY_H_
#define ONEFLOW_CORE_UNARY_FUNC_NDARRAY_H_

namespace oneflow {

template<typename T, const T (*unary_func)(const T), typename X>
class XpuUnaryFuncNdarray final {
 public:
  OF_DEVICE_FUNC XpuUnaryFuncNdarray(const X& x) : x_(x) {}

  template<int NDIMS>
  OF_DEVICE_FUNC T Get(int64_t offset) const {
    return unary_func(x_.Get<NDIMS>(offset));
  }

 private:
  const X& x_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_UNARY_FUNC_NDARRAY_H_
