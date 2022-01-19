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
#include "oneflow/user/kernels/categorical_ordinal_encode_kernel_util.h"

namespace oneflow {

template<typename T>
struct CategoricalOrdinalEncodeKernelUtil<DeviceType::kCPU, T> {
  static void Encode(ep::Stream* stream, int64_t capacity, T* table, T* size, int64_t n,
                     const T* hash, T* out) {
    for (int64_t i = 0; i < n; ++i) {
      const T h = hash[i];
      bool success = false;
      for (int64_t count = 0; count < capacity; ++count) {
        size_t idx =
            (static_cast<size_t>(h) + static_cast<size_t>(count)) % static_cast<size_t>(capacity);
        T* k_ptr = table + idx * 2;
        T* v_ptr = k_ptr + 1;
        if (*k_ptr == h) {
          out[i] = *v_ptr;
          success = true;
          break;
        } else if (*k_ptr == 0) {
          T new_size = *size + 1;
          *k_ptr = h;
          *v_ptr = new_size;
          out[i] = new_size;
          *size = new_size;
          success = true;
          break;
        } else {
          continue;
        }
      }
      CHECK(success);
    }
  }
};

#define INSTANTIATE_CATEGORICAL_ORDINAL_ENCODE_KERNEL_UTIL_CPU(type_cpp, type_proto) \
  template struct CategoricalOrdinalEncodeKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CATEGORICAL_ORDINAL_ENCODE_KERNEL_UTIL_CPU, INDEX_DATA_TYPE_SEQ);
#undef INSTANTIATE_CATEGORICAL_ORDINAL_ENCODE_KERNEL_UTIL_CPU

}  // namespace oneflow
