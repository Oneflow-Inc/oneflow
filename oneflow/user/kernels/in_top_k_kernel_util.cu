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
#include "oneflow/user/kernels/in_top_k_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void InTopkGpu(const int instance_num, const int classes_num, const T* targets,
                          const float* predictions, const int k, int8_t* out) {
  CUDA_1D_KERNEL_LOOP(idx, instance_num) {
    T target = targets[idx];
    bool cannot_say = (target >= classes_num) || !isfinite(predictions[idx * classes_num + target]);

    int32_t more_probable_classes = 0;
    if (!cannot_say) {
      const float target_prediction = predictions[idx * classes_num + target];
      FOR_RANGE(int32_t, class_idx, 0, classes_num) {
        float pred = predictions[idx * classes_num + class_idx];

        if (!isfinite(pred)) {
          cannot_say = true;
          break;
        } else if (pred > target_prediction) {
          ++more_probable_classes;
          if (more_probable_classes > k) break;
        }
      }
    }
    out[idx] = cannot_say ? false : (more_probable_classes < k);
  }
}

}  // namespace

template<typename T>
struct InTopkKernelUtil<DeviceType::kGPU, T> {
  static void InTopk(DeviceCtx* ctx, const int instance_num, const int classes_num,
                     const T* targets, const float* predictions, const int k, int8_t* out) {
    RUN_CUDA_KERNEL((InTopkGpu<T>), ctx, instance_num, instance_num, classes_num, targets,
                    predictions, k, out);
  }
};

#define INSTANTIATE_IN_TOP_K_KERNEL_UTIL_GPU(cpp_data_type, data_type) \
  template struct InTopkKernelUtil<DeviceType::kGPU, cpp_data_type>;

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_IN_TOP_K_KERNEL_UTIL_GPU, INDEX_DATA_TYPE_SEQ)

#undef INSTANTIATE_IN_TOP_K_KERNEL_UTIL_GPU

}  // namespace oneflow
