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
#ifndef ONEFLOW_CORE_COMMON_STREAM_ROLE_H_
#define ONEFLOW_CORE_COMMON_STREAM_ROLE_H_

#include <functional>
#include <array>
#include "oneflow/core/common/preprocessor.h"
#include "glog/logging.h"

namespace oneflow {

enum class StreamRole {
  kInvalid = 0,
  kCompute,
  kHost2Device,
  kDevice2Host,
  kSyncedLaunchedCommNet,
  kAsyncedLaunchedCommNet,
  kBarrier,
  kCriticalSection,
  kLazyJobLauncher,
  kPinnedCompute
};

template<typename DerivedT>
struct StreamRoleVisitor {
  template<typename... Args>
  static auto Visit(StreamRole stream_role, Args&&... args) {
    switch (stream_role) {
      case StreamRole::kInvalid: LOG(FATAL) << "invalid stream role";
      case StreamRole::kCompute: return DerivedT::VisitCompute(std::forward<Args>(args)...);
      case StreamRole::kHost2Device: return DerivedT::VisitHost2Device(std::forward<Args>(args)...);
      case StreamRole::kDevice2Host: return DerivedT::VisitDevice2Host(std::forward<Args>(args)...);
      case StreamRole::kSyncedLaunchedCommNet:
        return DerivedT::VisitSyncedLaunchedCommNet(std::forward<Args>(args)...);
      case StreamRole::kAsyncedLaunchedCommNet:
        return DerivedT::VisitAsyncedLaunchedCommNet(std::forward<Args>(args)...);
      case StreamRole::kBarrier: return DerivedT::VisitBarrier(std::forward<Args>(args)...);
      case StreamRole::kCriticalSection:
        return DerivedT::VisitCriticalSection(std::forward<Args>(args)...);
      case StreamRole::kLazyJobLauncher:
        return DerivedT::VisitLazyJobLauncher(std::forward<Args>(args)...);
      case StreamRole::kPinnedCompute:
        return DerivedT::VisitPinnedCompute(std::forward<Args>(args)...);
    }
    LOG(FATAL) << "invalid stream role";
  }
};

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::StreamRole> final {
  size_t operator()(const oneflow::StreamRole& stream_role) const {
    return static_cast<int>(stream_role);
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_COMMON_STREAM_ROLE_H_
