#ifndef ONEFLOW_CORE_FRAMEWORK_STREAM_ON_INDEPENDENT_THREAD_H_
#define ONEFLOW_CORE_FRAMEWORK_STREAM_ON_INDEPENDENT_THREAD_H_

#include <glog/logging.h>
#include "oneflow/core/common/stream_role.h"

namespace oneflow {

struct StreamOnIndependentThread {
  static bool Case(StreamRoleCase<StreamRole::kInvalid>) {  // NOLINT
    LOG(FATAL);
  }
  static bool Case(StreamRoleCase<StreamRole::kCompute>) { return false; }
  static bool Case(StreamRoleCase<StreamRole::kHost2Device>) { return false; }
  static bool Case(StreamRoleCase<StreamRole::kDevice2Host>) { return false; }
  static bool Case(StreamRoleCase<StreamRole::kSyncedLaunchedCommNet>) { return false; }
  static bool Case(StreamRoleCase<StreamRole::kAsyncedLaunchedCommNet>) { return false; }
  static bool Case(StreamRoleCase<StreamRole::kBarrier>) { return false; }
  static bool Case(StreamRoleCase<StreamRole::kCriticalSection>) { return false; }
  static bool Case(StreamRoleCase<StreamRole::kLazyJobLauncher>) { return true; }
};

}

#endif  // ONEFLOW_CORE_FRAMEWORK_STREAM_ON_INDEPENDENT_THREAD_H_
