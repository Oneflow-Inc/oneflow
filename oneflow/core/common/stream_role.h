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

namespace oneflow {

#define STREAM_ROLE_SEQ                         \
  OF_PP_MAKE_TUPLE_SEQ(kCompute)                \
  OF_PP_MAKE_TUPLE_SEQ(kHost2Device)            \
  OF_PP_MAKE_TUPLE_SEQ(kDevice2Host)            \
  OF_PP_MAKE_TUPLE_SEQ(kSyncedLaunchedCommNet)  \
  OF_PP_MAKE_TUPLE_SEQ(kAsyncedLaunchedCommNet) \
  OF_PP_MAKE_TUPLE_SEQ(kCriticalSection)

enum class StreamRole {
  kInvalid = 0,
#define DECLARE_STREAM_ROLE(stream_role) stream_role,
  OF_PP_FOR_EACH_TUPLE(DECLARE_STREAM_ROLE, STREAM_ROLE_SEQ)
#undef DECLARE_STREAM_ROLE
};

static constexpr int kStreamRoleSize = 1 + OF_PP_SEQ_SIZE(STREAM_ROLE_SEQ);

// Act as a class for overloading functions
template<StreamRole stream_role>
struct StreamRoleCase {};

template<typename Functor, typename... Args>
auto StreamRoleSwitch(StreamRole stream_role, Args&&... args)
    -> decltype(Functor::Case(StreamRoleCase<StreamRole::kInvalid>(),
                              std::forward<Args>(args)...)) {
  switch (stream_role) {
#define MAKE_ENTRY(stream_role) \
  case StreamRole::stream_role: \
    return Functor::Case(StreamRoleCase<StreamRole::stream_role>(), std::forward<Args>(args)...);
    OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, STREAM_ROLE_SEQ)
#undef MAKE_ENTRY
    default:
      return Functor::Case(StreamRoleCase<StreamRole::kInvalid>(), std::forward<Args>(args)...);
  }
}

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
