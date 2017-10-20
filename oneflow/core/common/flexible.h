#ifndef ONEFLOW_CORE_COMMON_FLEXIBLE_H_
#define ONEFLOW_CORE_COMMON_FLEXIBLE_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename flexible_struct>
size_t FlexibleSizeOf(uint32_t n) {
  return sizeof(flexible_struct);
}

template<typename flexible_struct>
size_t FlexibleSizeOf(const flexible_struct& obj) {
  return sizeof(flexible_struct);
}

template<typename flexible_struct>
void FlexibleSetArraySize(flexible_struct* type, size_t len) {}

template<typename T>
static std::unique_ptr<T, decltype(&free)> FlexibleMalloc(size_t len) {
  T* ptr = reinterpret_cast<T*>(malloc(FlexibleSizeOf<T>(len)));
  FlexibleSetArraySize(ptr, len);
  return std::unique_ptr<T, decltype(&free)>(ptr, &free);
}

template<typename T>
static std::unique_ptr<T, decltype(&free)> FlexibleMalloc() {
  return FlexibleMalloc<T>(0);
}

}  // namespace oneflow
#endif  // ONEFLOW_CORE_COMMON_FLEXIBLE_H_
