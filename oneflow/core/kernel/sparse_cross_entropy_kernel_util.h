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
#ifndef ONEFLOW_CORE_KERNEL_SPARSE_CROSS_ENTROPY_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_SPARSE_CROSS_ENTROPY_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
struct SparseCrossEntropyKernelUtil {
  static void ComputeEntropy(DeviceCtx* ctx, int64_t num_instances, int64_t num_classes, const T* x,
                             const K* labels, T* y);
  static void ComputeDiff(DeviceCtx* ctx, int64_t num_instances, int64_t num_classes, const T* x,
                          const K* labels, T* dx);
  static void ComputeDiff(DeviceCtx* ctx, int64_t num_instances, int64_t num_classes, const T* x,
                          const K* labels, const T* dy, T* dx);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SPARSE_CROSS_ENTROPY_KERNEL_UTIL_H_
