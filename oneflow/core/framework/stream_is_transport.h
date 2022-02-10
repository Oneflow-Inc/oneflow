#ifndef ONEFLOW_CORE_FRAMEWORK_STREAM_IS_TRANSPORT_H_
#define ONEFLOW_CORE_FRAMEWORK_STREAM_IS_TRANSPORT_H_

#include <glog/logging.h>
#include "oneflow/core/common/stream_role.h"

namespace oneflow {

struct StreamIsTransport {
  static bool Call(SRCase<StreamRole::kInvalid>) {  // NOLINT
    LOG(FATAL);
  }
  static bool Call(SRCase<StreamRole::kCompute>) { return false; }
  static bool Call(SRCase<StreamRole::kHost2Device>) { return false; }
  static bool Call(SRCase<StreamRole::kDevice2Host>) { return false; }
  static bool Call(SRCase<StreamRole::kSyncedLaunchedCC>) { return true; }
  static bool Call(SRCase<StreamRole::kAsyncedLaunchedCC>) { return true; }
  static bool Call(SRCase<StreamRole::kCriticalSection>) { return false; }
};

}

#endif  // ONEFLOW_CORE_FRAMEWORK_STREAM_IS_TRANSPORT_H_
