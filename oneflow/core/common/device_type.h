/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_COMMON_DEVICE_TYPE_H_
#define ONEFLOW_CORE_COMMON_DEVICE_TYPE_H_

#include "glog/logging.h"
#include "oneflow/core/common/device_type.pb.h"

namespace oneflow {

template<typename DerivedT>
struct DeviceTypeVisitor {
  template<typename... Args>
  static auto Visit(DeviceType device_type, Args&&... args) {
    switch (device_type) {
      case DeviceType::kInvalidDevice: LOG(FATAL) << "invalid device type";
      case DeviceType::kCPU: return DerivedT::VisitCPU(std::forward<Args>(args)...);
      case DeviceType::kCUDA: return DerivedT::VisitCUDA(std::forward<Args>(args)...);
      case DeviceType::kMockDevice: return DerivedT::VisitMockDevice(std::forward<Args>(args)...);
    }
    LOG(FATAL) << "invalid device type";
  }
};

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::DeviceType> final {
  size_t operator()(oneflow::DeviceType device_type) const {
    return static_cast<size_t>(device_type);
  }
};

}  // namespace std

namespace oneflow {

inline std::string PrintAvailableDevices() {
  std::string str("cpu");
#ifdef WITH_CUDA
  str += ", cuda";
#endif
  return str;
}

inline std::string PrintGeneratorAvailableDevices() {
  std::string str("cpu");
#ifdef WITH_CUDA
  str += ", cuda";
#endif
  str += ", auto";  // "auto" is a fake device type for random generator.
  return str;
}

#if defined(WITH_CUDA)
#define DEVICE_TYPE_SEQ                  \
  OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCPU) \
  OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCUDA)
#else
#define DEVICE_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCPU)
#endif

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_DEVICE_TYPE_H_
