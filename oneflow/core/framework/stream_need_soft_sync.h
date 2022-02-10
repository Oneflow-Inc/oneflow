#ifndef ONEFLOW_CORE_FRAMEWORK_STREAM_NEED_SOFT_SYNC_H_
#define ONEFLOW_CORE_FRAMEWORK_STREAM_NEED_SOFT_SYNC_H_

#include <glog/logging.h>
#include "oneflow/core/common/stream_role.h"

namespace oneflow {

struct NeedSoftSync {
  static bool Case(SR<StreamRole::kInvalid>) {  // NOLINT
    LOG(FATAL);
  }
  static bool Case(SR<StreamRole::kCompute>) { return true; }
  static bool Case(SR<StreamRole::kHost2Device>) { return false; }
  static bool Case(SR<StreamRole::kDevice2Host>) { return false; }
  static bool Case(SR<StreamRole::kSyncedLaunchedCC>) { return true; }
  static bool Case(SR<StreamRole::kAsyncedLaunchedCC>) { return false; }
  static bool Case(SR<StreamRole::kCriticalSection>) { return false; }
};

}

#endif  // ONEFLOW_CORE_FRAMEWORK_STREAM_NEED_SOFT_SYNC_H_
