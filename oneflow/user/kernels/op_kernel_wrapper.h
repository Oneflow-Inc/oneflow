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
#ifndef ONEFLOW_USER_KERNELS_OP_KERNEL_STATE_WRAPPER_H_
#define ONEFLOW_USER_KERNELS_OP_KERNEL_STATE_WRAPPER_H_

#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

template<typename T>
class OpKernelStateWrapper final : public user_op::OpKernelState {
 public:
  template<typename... Args>
  explicit OpKernelStateWrapper(Args&&... args) : data_(std::forward<Args>(args)...) {}

  ~OpKernelStateWrapper() = default;

  const T& Get() const { return data_; }
  T* Mutable() { return &data_; }

 private:
  T data_;
};

template<typename T>
class OpKernelCacheWrapper final : public user_op::OpKernelCache {
 public:
  template<typename... Args>
  explicit OpKernelCacheWrapper(Args&&... args) : data_(std::forward<Args>(args)...) {}

  ~OpKernelCacheWrapper() = default;

  const T& Get() const { return data_; }
  T* Mutable() { return &data_; }

 private:
  T data_;
};

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_OP_KERNEL_STATE_WRAPPER_H_
