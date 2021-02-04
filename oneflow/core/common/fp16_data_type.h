#ifndef ONEFLOW_CORE_COMMON_FP16_DATA_TYPE_H_
#define ONEFLOW_CORE_COMMON_FP16_DATA_TYPE_H_
#include <type_traits>

// TODO: auto generated
#include "oneflow/core/framework/device_register_gpu.h"

namespace oneflow {
// Type Trait: IsFloat16
template<typename T>
struct IsFloat16 : std::integral_constant<bool, false> {};
}

#endif  // ONEFLOW_CORE_COMMON_FP16_DATA_TYPE_H_
