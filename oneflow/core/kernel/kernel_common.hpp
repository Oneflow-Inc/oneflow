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
#ifndef ONEFLOW_CORE_KERNEL_KERNEL_COMMON_HPP_
#define ONEFLOW_CORE_KERNEL_KERNEL_COMMON_HPP_

#include "oneflow/core/common/util.h"

namespace oneflow {
// function object for add/clone kernel
template<DeviceType device_type, typename T, typename... Args>
inline std::enable_if_t<std::is_same<T, float>::value> Addition(DeviceCtx* device_ctx, Blob* out,
                                                                Args... in) {
  KernelUtil<device_type, float>::Addition(device_ctx, out->shape().elem_cnt(),
                                           out->mut_dptr<float>(), in->template dptr<float>()...);
}
template<DeviceType device_type, typename T, typename... Args>
inline std::enable_if_t<std::is_same<T, double>::value> Addition(DeviceCtx* device_ctx, Blob* out,
                                                                 Args... in) {
  KernelUtil<device_type, double>::Addition(
      device_ctx, out->shape().elem_cnt(), out->mut_dptr<double>(), in->template dptr<double>()...);
}

template<DeviceType device_type, typename T, typename... Args>
inline void Addition(...) {
  LOG(FATAL) << "just support float point here";
}

template<DeviceType device_type, typename T, typename U>
struct AdditionFunction {
  template<typename V>
  void operator()(V v) {
    AdditionImpl(std::make_index_sequence<decltype(v)::value>());
  }

  template<size_t... Idx>
  void AdditionImpl(std::index_sequence<Idx...>) {
    Addition<device_type, T>(device_ctx_, diff_blob_,
                             BnInOp2Blob_(u_->op_attribute().input_bns(offset_ + Idx))...);
  }

  void AdditionImpl(std::index_sequence<>) {}

  Blob* diff_blob_;
  std::function<Blob*(const std::string&)> BnInOp2Blob_;
  DeviceCtx* device_ctx_;
  int32_t offset_;
  U u_;
};
}  // namespace oneflow
#endif  // ONEFLOW_CORE_KERNEL_KERNEL_COMMON_HPP_
