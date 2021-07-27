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
#ifndef ONEFLOW_CORE_DEVICE_HIP_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_HIP_DEVICE_CONTEXT_H_

#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/device/hip_stream_handle.hip.h"

namespace oneflow {

#ifdef WITH_HIP

class HipDeviceCtx : public DeviceCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(HipDeviceCtx);
  HipDeviceCtx() = delete;
  ~HipDeviceCtx() override = default;

  explicit HipDeviceCtx(HipStreamHandle* hip_handler) : hip_handler_(hip_handler) {}

  const hipStream_t& hip_stream() const override { return *(hip_handler_->hip_stream()); }

  const hipblasHandle_t& hipblas_pmh_handle() const override {
    return *(hip_handler_->hipblas_pmh_handle());
  }
  const hipblasHandle_t& hipblas_tensor_op_math_handle() const override {
    return *(hip_handler_->hipblas_tensor_op_math_handle());
  }
  const hipblasHandle_t& hipblas_pmd_handle() const override {
    return *(hip_handler_->hipblas_pmd_handle());
  }

  const miopenHandle_t& miopen_handle() const override { return *(hip_handler_->miopen_handle()); }
  
  void SyncDevice() override { OF_HIP_CHECK(hipStreamSynchronize(hip_stream())); }

  void AddCallBack(std::function<void()> callback) const override {
    hip_handler_->AddCallBack(callback);
  }

 protected:
  HipStreamHandle* hip_handler_;
};

#endif  // WITH_HIP

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_HIP_DEVICE_CONTEXT_H_
