#ifndef ONEFLOW_CORE_COMMON_FLEXIBLE_H_
#define ONEFLOW_CORE_COMMON_FLEXIBLE_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename T>
class Flexible final {
 public:
  static std::unique_ptr<T, decltype(&free)> Malloc(size_t array_size);
  static size_t SizeOf(size_t n);
  static size_t SizeOf(const T& obj);
  static void SetArraySize(T* obj, size_t array_size);
  static std::unique_ptr<T, decltype(&free)> Malloc() { return Malloc(0); }
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_COMMON_FLEXIBLE_H_
