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
#ifndef ONEFLOW_CORE_KERNEL_PACK_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_PACK_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type>
class PackKernelUtil final {
 public:
  static void Pack(DeviceCtx* ctx, size_t in_index, size_t total_pack_num, const Blob* in_blob,
                   Blob* out_blob);
  static void Unpack(DeviceCtx* ctx, size_t out_index, size_t total_unpack_num, const Blob* in_blob,
                     Blob* out_blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PACK_KERNEL_UTIL_H_
