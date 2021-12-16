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

#include "oneflow/core/framework/config_dtr.h"

namespace oneflow {

Maybe<void> EnableDTRStrategy(bool enable_dtr, size_t thres, bool enable_debug, int memory_policy,
                              bool use_disjoint_set) {
  CHECK_NOTNULL_OR_RETURN((Global<DTRConfig>::Get()));
  *Global<DTRConfig>::Get() =
      DTRConfig(enable_dtr, thres, enable_debug, memory_policy, use_disjoint_set);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
