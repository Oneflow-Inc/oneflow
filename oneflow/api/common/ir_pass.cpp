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

#ifdef WITH_MLIR

#include "oneflow/ir/include/OneFlow/Extension.h"
#include "oneflow/ir/oneflow-extension/include/OneFlow/OneFlowRoundTrip.h"
#include <glog/logging.h>

namespace oneflow {

REGISTER_JOB_PASS("IRRoundTripBeforeAD", IRRoundTrip<kBeforeAD>);
REGISTER_JOB_PASS("IRRoundTrip", IRRoundTrip<kAfterAD>);

}  // namespace oneflow

#endif  // WITH_MLIR
