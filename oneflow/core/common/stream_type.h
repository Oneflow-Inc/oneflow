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
#ifndef ONEFLOW_CORE_COMMON_STREAM_TYPE_H_
#define ONEFLOW_CORE_COMMON_STREAM_TYPE_H_

#include <functional>
#include <array>
#include "oneflow/core/common/preprocessor.h"
#include "glog/logging.h"

namespace oneflow {

enum class StreamType {
  kInvalid = 0,
  kCompute,
  kHost2Device,
  kDevice2Host,
  kCcl,
  kBarrier,
  kCriticalSection,
  kLazyJobLauncher,
  kPinnedCompute
};

template<typename DerivedT>
struct StreamTypeVisitor {
  template<typename... Args>
  static auto Visit(StreamType stream_type, Args&&... args) {
    switch (stream_type) {
      case StreamType::kInvalid: LOG(FATAL) << "invalid stream type";
      case StreamType::kCompute: return DerivedT::VisitCompute(std::forward<Args>(args)...);
      case StreamType::kHost2Device: return DerivedT::VisitHost2Device(std::forward<Args>(args)...);
      case StreamType::kDevice2Host: return DerivedT::VisitDevice2Host(std::forward<Args>(args)...);
      case StreamType::kCcl: return DerivedT::VisitCcl(std::forward<Args>(args)...);
      case StreamType::kBarrier: return DerivedT::VisitBarrier(std::forward<Args>(args)...);
      case StreamType::kCriticalSection:
        return DerivedT::VisitCriticalSection(std::forward<Args>(args)...);
      case StreamType::kLazyJobLauncher:
        return DerivedT::VisitLazyJobLauncher(std::forward<Args>(args)...);
      case StreamType::kPinnedCompute:
        return DerivedT::VisitPinnedCompute(std::forward<Args>(args)...);
    }
    LOG(FATAL) << "invalid stream type";
  }
};

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::StreamType> final {
  size_t operator()(const oneflow::StreamType& stream_type) const {
    return static_cast<int>(stream_type);
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_COMMON_STREAM_TYPE_H_
