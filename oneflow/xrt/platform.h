#ifndef ONEFLOW_XRT_PLATFORM_H_
#define ONEFLOW_XRT_PLATFORM_H_

#include "oneflow/xrt/platform.pb.h"

namespace oneflow {
namespace xrt {
// NOLINT
}  // namespace xrt
}  // namespace oneflow

namespace std {

template <>
struct hash<oneflow::xrt::XrtDevice> {
  size_t operator()(const oneflow::xrt::XrtDevice &device) const {
    return std::hash<int64_t>()(static_cast<int64_t>(device));
  }
};

template <>
struct hash<oneflow::xrt::XrtEngine> {
  size_t operator()(const oneflow::xrt::XrtEngine &engine) const {
    return std::hash<int64_t>()(static_cast<int64_t>(engine));
  }
};

}  // namespace std

#endif  // ONEFLOW_XRT_PLATFORM_H_
