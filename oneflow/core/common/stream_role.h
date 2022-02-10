#ifndef ONEFLOW_CORE_COMMON_STREAM_ROLE_H_
#define ONEFLOW_CORE_COMMON_STREAM_ROLE_H_

#include <hash>
#include <array>
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

#define STREAM_ROLE_SEQ                  \
  OF_PP_MAKE_TUPLE_SEQ(kCompute) \
  OF_PP_MAKE_TUPLE_SEQ(kHost2Device) \
  OF_PP_MAKE_TUPLE_SEQ(kDevice2Host) \
  OF_PP_MAKE_TUPLE_SEQ(kSyncedLaunchedCC) \
  OF_PP_MAKE_TUPLE_SEQ(kAsyncedLaunchedCC) \
  OF_PP_MAKE_TUPLE_SEQ(kCriticalSection)

enum StreamRole {
  kInvalid = 0,
#define DECLARE_STREAM_ROLE(stream_role) stream_role,
  OF_PP_FOR_EACH_TUPLE(DECLARE_STREAM_ROLE, STREAM_ROLE_SEQ)
#undef DECLARE_STREAM_ROLE
};

static constexpr int kStreamRoleSize = 1 + OF_PP_SEQ_SIZE(STREAM_ROLE_SEQ)

// stream role case
template<StreamRole stream_role>
struct SRCase { };

template<typename Functor, typename... Args>
auto SwitchCall(StreamRole stream_role, Args&&... args) -> decltype(Functor::Call(SRCase<StreamRole::kInvalid>(), std::forward<Args>(args)...)) {
  switch (stream_role) {
#define MAKE_ENTRY(stream_role) \
  case StreamRole::stream_role: \
    return Functor::Call(SRCase<StreamRole::stream_role>(), std::forward<Args>(args)...);
    OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, STREAM_ROLE_SEQ)
#undef MAKE_ENTRY
  default: return Functor::Call(SRCase<StreamRole::kInvalid>(), std::forward<Args>(args)...);
  }
}

}

namespace std {
template<>
struct hash<oneflow::StreamRole> final {
  size_t operator()(const oneflow::StreamRole& stream_role) const {
    return static_cast<int>(stream_role); 
  }
};

#endif  // ONEFLOW_CORE_COMMON_STREAM_ROLE_H_
