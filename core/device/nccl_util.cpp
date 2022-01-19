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
#include "oneflow/core/device/nccl_util.h"

namespace oneflow {

#ifdef WITH_CUDA

std::string NcclUniqueIdToString(const ncclUniqueId& unique_id) {
  return std::string(unique_id.internal, NCCL_UNIQUE_ID_BYTES);
}

void NcclUniqueIdFromString(const std::string& str, ncclUniqueId* unique_id) {
  CHECK_EQ(str.size(), NCCL_UNIQUE_ID_BYTES);
  memcpy(unique_id->internal, str.data(), NCCL_UNIQUE_ID_BYTES);
}

#endif  // WITH_CUDA

}  // namespace oneflow
