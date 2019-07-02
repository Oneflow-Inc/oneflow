#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_BACKEND_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_BACKEND_H_

#include <string>
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {
namespace mola {

template <DeviceType device_type>
struct Backend {
  static std::string to_string() {
    return "";
  }
};

template <>
struct Backend<kCPU> {
  static std::string to_string() {
    return "CPU";
  }
};

template <>
struct Backend<kGPU> {
  static std::string to_string() {
    return "CUDA";
  }
};

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_BACKEND_H_
