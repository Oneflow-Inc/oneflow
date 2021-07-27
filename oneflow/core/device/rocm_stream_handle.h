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
#ifndef ONEFLOW_CORE_DEVICE_ROCM_STREAM_HANDLE_H_
#define ONEFLOW_CORE_DEVICE_ROCM_STREAM_HANDLE_H_

#include "oneflow/core/common/channel.h"
#include "oneflow/core/device/hip_util.hip.h"

namespace oneflow {

#ifdef WITH_HIP

struct RocmCBEvent {
  std::function<void()> callback;
  hipEvent_t event;
};

class RocmStreamHandle final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RocmStreamHandle);
  RocmStreamHandle() = delete;
  RocmStreamHandle(Channel<RocmCBEvent>* cb_event_chan) : cb_event_chan_(cb_event_chan) {}

  const hipStream_t* rocm_stream();
  const hipblasHandle_t* hipblas_pmh_handle();
  const hipblasHandle_t* hipblas_pmd_handle();
  const hipblasHandle_t* hipblas_tensor_op_math_handle();
  const miopenHandle_t* miopen_handle();

  void AddCallBack(std::function<void()> callback);

  ~RocmStreamHandle();

 private:
  Channel<RocmCBEvent>* cb_event_chan_;
  std::unique_ptr<hipStream_t> rocm_stream_;
  std::unique_ptr<hipblasHandle_t> hipblas_pmh_handle_;
  std::unique_ptr<hipblasHandle_t> hipblas_pmd_handle_;
  std::unique_ptr<hipblasHandle_t> hipblas_tensor_op_math_handle_;
  std::unique_ptr<miopenHandle_t> miopen_handle_;
};

#endif  // WITH_HIP

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_ROCM_STREAM_HANDLE_H_
