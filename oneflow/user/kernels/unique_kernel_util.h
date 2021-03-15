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
#ifndef ONEFLOW_USER_KERNELS_UNIQUE_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_UNIQUE_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename KEY, typename IDX>
struct UniqueKernelUtil {
  static void Unique(DeviceCtx* ctx, int64_t n, const KEY* in, IDX* num_unique, KEY* unique_out,
                     IDX* idx_out, void* workspace, int64_t workspace_size_in_bytes);
  static void UniqueWithCounts(DeviceCtx* ctx, int64_t n, const KEY* in, IDX* num_unique,
                               KEY* unique_out, IDX* idx_out, IDX* count, void* workspace,
                               int64_t workspace_size_in_bytes);
  static void GetUniqueWorkspaceSizeInBytes(DeviceCtx* ctx, int64_t n,
                                            int64_t* workspace_size_in_bytes);
  static void GetUniqueWithCountsWorkspaceSizeInBytes(DeviceCtx* ctx, int64_t n,
                                                      int64_t* workspace_size_in_bytes);
};

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_UNIQUE_KERNEL_UTIL_H_
