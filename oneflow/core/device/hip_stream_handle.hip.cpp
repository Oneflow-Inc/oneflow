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
#include "oneflow/core/device/hip_stream_handle.hip.h"
#include "oneflow/core/device/hip_util.hip.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"

namespace oneflow {

#ifdef WITH_HIP

const hipStream_t* HipStreamHandle::hip_stream() {
  if (!hip_stream_) {
    hip_stream_.reset(new hipStream_t);
    OF_HIP_CHECK(hipStreamCreate(hip_stream_.get()));
  }
  return hip_stream_.get();
}

const hipblasHandle_t* HipStreamHandle::hipblas_pmh_handle() {
  if (!hipblas_pmh_handle_) {
    hipblas_pmh_handle_.reset(new hipblasHandle_t);
    OF_HIPBLAS_CHECK(hipblasCreate(hipblas_pmh_handle_.get()));
    OF_HIPBLAS_CHECK(hipblasSetStream(*hipblas_pmh_handle_, *hip_stream()));
  }
  return hipblas_pmh_handle_.get();
}

const hipblasHandle_t* HipStreamHandle::hipblas_pmd_handle() {
  if (!hipblas_pmd_handle_) {
    hipblas_pmd_handle_.reset(new hipblasHandle_t);
    OF_HIPBLAS_CHECK(hipblasCreate(hipblas_pmd_handle_.get()));
    OF_HIPBLAS_CHECK(hipblasSetStream(*hipblas_pmd_handle_, *hip_stream()));
    OF_HIPBLAS_CHECK(hipblasSetPointerMode(*hipblas_pmd_handle_, HIPBLAS_POINTER_MODE_DEVICE));
  }
  return hipblas_pmd_handle_.get();
}

const hipblasHandle_t* HipStreamHandle::hipblas_tensor_op_math_handle() {
  if (!hipblas_tensor_op_math_handle_) {
    hipblas_tensor_op_math_handle_.reset(new hipblasHandle_t);
    OF_HIPBLAS_CHECK(hipblasCreate(hipblas_tensor_op_math_handle_.get()));
    OF_HIPBLAS_CHECK(hipblasSetStream(*hipblas_tensor_op_math_handle_, *hip_stream()));
  }
  return hipblas_tensor_op_math_handle_.get();
}

const miopenHandle_t* HipStreamHandle::miopen_handle() {
  if (!miopen_handle_) {
    OF_HIP_CHECK(hipDeviceSynchronize());
    OF_HIP_CHECK(hipGetLastError());
    miopen_handle_.reset(new miopenHandle_t);
    OF_MIOPEN_CHECK(miopenCreate(miopen_handle_.get()));
    OF_HIP_CHECK(hipDeviceSynchronize());
    hipGetLastError();
    OF_MIOPEN_CHECK(miopenSetStream(*miopen_handle_, *hip_stream()));
  }
  return miopen_handle_.get();
}

void HipStreamHandle::AddCallBack(std::function<void()> callback) {
  HipCBEvent cb_event;
  cb_event.callback = std::move(callback);
  OF_HIP_CHECK(hipEventCreateWithFlags(&(cb_event.event), hipEventDisableTiming));
  OF_HIP_CHECK(hipEventRecord(cb_event.event, *hip_stream()));
  cb_event_chan_->Send(cb_event);
}

HipStreamHandle::~HipStreamHandle() {
  if (hip_stream_) { OF_HIP_CHECK(hipStreamDestroy(*hip_stream_)); }
}

#endif  // WITH_HIP

}  // namespace oneflow
