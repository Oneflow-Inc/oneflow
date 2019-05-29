#ifndef ONEFLOW_CORE_NDARRAY_XPU_BINARY_FUNC_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_XPU_BINARY_FUNC_NDARRAY_H_

namespace oneflow {

template<typename T, template<typename> class binary_func, typename A, typename B>
class XpuBinaryFuncNdarray final {
 public:
  OF_DEVICE_FUNC XpuBinaryFuncNdarray(const A& a, const B& b) : a_(a), b_(b) {}

  template<int NDIMS>
  OF_DEVICE_FUNC T Get(int64_t offset) const {
    return binary_func<T>::Invoke(a_.Get<NDIMS>(offset), b_.Get<NDIMS>(offset));
  }

 private:
  const A& a_;
  const B& b_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_BINARY_FUNC_NDARRAY_H_
