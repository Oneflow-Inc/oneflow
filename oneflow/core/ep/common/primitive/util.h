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
#ifndef ONEFLOW_CORE_EP_COMMON_PRIMITIVE_UTIL_H_
#define ONEFLOW_CORE_EP_COMMON_PRIMITIVE_UTIL_H_

namespace oneflow {

namespace ep {
namespace primitive {

inline size_t GetElementCount(size_t num_dims, const int64_t* dims) {
  size_t count = 1;
  for (size_t i = 0; i < num_dims; ++i) { count *= dims[i]; }
  return count;
}

template<typename T>
bool IsPackSizeSupported(const size_t pack_size, size_t num_dims, const int64_t* dims,
                         std::uintptr_t ptr) {
  if ((dims[num_dims - 1] % pack_size == 0) && (ptr % (pack_size * sizeof(T)) == 0)) {
    return true;
  } else {
    return false;
  }
};

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_COMMON_PRIMITIVE_UTIL_H_
