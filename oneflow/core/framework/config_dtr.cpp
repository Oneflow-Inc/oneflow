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

Maybe<void> EnableDTRStrategy(bool enable_dtr, double thres, bool enable_debug) {
  CHECK_NOTNULL_OR_RETURN((Global<bool, EnableDTR>::Get()));
  CHECK_NOTNULL_OR_RETURN((Global<double, DTRMemoryThreshold>::Get()));
  CHECK_NOTNULL_OR_RETURN((Global<size_t, DTRRemainMemory>::Get()));
  CHECK_NOTNULL_OR_RETURN((Global<bool, EnableDTRDebug>::Get()));
  *Global<bool, EnableDTR>::Get() = enable_dtr;
  *Global<double, DTRMemoryThreshold>::Get() = thres;
  *Global<bool, EnableDTRDebug>::Get() = enable_debug;
  size_t free_bytes = -1;
  size_t total_bytes = -1;
  OF_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  *Global<size_t, DTRRemainMemory>::Get() = (1 - thres) * free_bytes;
  return Maybe<void>::Ok();
}

}  // namespace oneflow
