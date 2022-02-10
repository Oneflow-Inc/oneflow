#ifndef ONEFLOW_CORE_FRAMEWORK_STREAM_IS_TRANSPORT_H_
#define ONEFLOW_CORE_FRAMEWORK_STREAM_IS_TRANSPORT_H_

#include <glog/logging.h>
#include "oneflow/core/common/stream_role.h"

namespace oneflow {

struct StreamIsTransport {
  static bool Case(SR<StreamRole::kInvalid>) {  // NOLINT
    LOG(FATAL);
  }
  static bool Case(SR<StreamRole::kCompute>) { return false; }
  static bool Case(SR<StreamRole::kHost2Device>) { return false; }
  static bool Case(SR<StreamRole::kDevice2Host>) { return false; }
  static bool Case(SR<StreamRole::kSyncedLaunchedCC>) { return true; }
  static bool Case(SR<StreamRole::kAsyncedLaunchedCC>) { return true; }
  static bool Case(SR<StreamRole::kCriticalSection>) { return false; }
};

}

#endif  // ONEFLOW_CORE_FRAMEWORK_STREAM_IS_TRANSPORT_H_
