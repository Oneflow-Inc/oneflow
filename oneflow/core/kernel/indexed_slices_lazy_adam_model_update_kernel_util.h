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
#ifndef ONEFLOW_CORE_KERNEL_INDEXED_SLICES_LAZY_ADAM_MODEL_UPDATE_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_INDEXED_SLICES_LAZY_ADAM_MODEL_UPDATE_KERNEL_UTIL_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K, typename IDX>
struct IndexedSlicesLazyAdamMdUpdateKernelUtil {
  static void Update(DeviceCtx* ctx, T beta1, T beta2, T epsilon, int64_t num_instance,
                     int64_t feature_size, int64_t lower_bound, int64_t upper_bound,
                     const IDX* num_unique_instance, const int64_t* train_step,
                     const float* learning_rate, const K* indices, const T* values, T* model, T* m,
                     T* v);
  static void ComputeLocalLearningRate(DeviceCtx* ctx, T beta1, T beta2, const int64_t* train_step,
                                       const float* learning_rate, float* local_learning_rate);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_INDEXED_SLICES_LAZY_ADAM_MODEL_UPDATE_KERNEL_UTIL_H_
