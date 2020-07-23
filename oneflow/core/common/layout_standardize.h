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
#ifndef ONEFLOW_CORE_COMMON_LAYOUT_STANDARDIZE_H_
#define ONEFLOW_CORE_COMMON_LAYOUT_STANDARDIZE_H_

namespace oneflow {

template<typename T>
class LayoutStandardize final {
 public:
  void __Init__(const T& val) { new (&data_[0]) T(val); }
  void __Delete__() { Mutable()->~T(); }

  const T& Get() const { return *reinterpret_cast<const T*>(&data_[0]); }
  T* Mutable() { return reinterpret_cast<T*>(&data_[0]); }

 private:
  union {
    char data_[sizeof(T)];
    int64_t align_;
  };
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_LAYOUT_STANDARDIZE_H_
