#include <glog/logging.h>
#include "oneflow/core/common/global.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/multi_client.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/control/ctrl_bootstrap.pb.h"
#include "oneflow/core/framework/shut_down_util.h"

namespace oneflow_api {

inline void StartOneFlow() {
  oneflow::Global<oneflow::ProcessCtx>::New();
  CHECK_JUST(oneflow::SetIsMultiClient(false));
}

inline void FinalizeOneFlow() {
  oneflow::SetShuttingDown();
}

} // namespace oneflow_api
