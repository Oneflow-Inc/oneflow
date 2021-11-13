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
