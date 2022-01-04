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
#include "oneflow/core/boxing/eager_boxing_logger.h"
#include "oneflow/core/boxing/boxing_interpreter_status.h"

namespace oneflow {

void NaiveEagerBoxingLogger::Log(const BoxingInterpreterStatus& status) {
  LOG(INFO) << "boxing interpreter route: " << (status.boxing_desc());
  LOG(INFO) << "Altered state of sbp: " << (status.nd_sbp_routing());
  LOG(INFO) << "Altered state placement: " << (status.placement_routing());
}

}  // namespace oneflow
