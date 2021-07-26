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
#include "oneflow/core/device/rocm_stream_handle.h"
#include "oneflow/core/device/rocm_util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"

namespace oneflow {

#ifdef WITH_HIP

const hipStream_t* RocmStreamHandle::rocm_stream() {
  if (!rocm_stream_) {
    rocm_stream_.reset(new hipStream_t);
    OF_ROCM_CHECK(hipStreamCreate(rocm_stream_.get()));
  }
  return rocm_stream_.get();
}

const hipblasHandle_t* RocmStreamHandle::hipblas_pmh_handle() {
  if (!hipblas_pmh_handle_) {
    hipblas_pmh_handle_.reset(new hipblasHandle_t);
    OF_HIPBLAS_CHECK(hipblasCreate(hipblas_pmh_handle_.get()));
    OF_HIPBLAS_CHECK(hipblasSetStream(*hipblas_pmh_handle_, *rocm_stream()));
// #if CUDA_VERSION >= 11000
//     if (Global<ResourceDesc, ForSession>::Get()->enable_tensor_float_32_compute()) {
//       OF_HIPBLAS_CHECK(cublasSetMathMode(*cublas_pmh_handle_, CUBLAS_TF32_TENSOR_OP_MATH));
//     }
// #endif
  }
  return hipblas_pmh_handle_.get();
}

const hipblasHandle_t* RocmStreamHandle::hipblas_pmd_handle() {
  if (!hipblas_pmd_handle_) {
    hipblas_pmd_handle_.reset(new hipblasHandle_t);
    OF_HIPBLAS_CHECK(hipblasCreate(hipblas_pmd_handle_.get()));
    OF_HIPBLAS_CHECK(hipblasSetStream(*hipblas_pmd_handle_, *rocm_stream()));
    OF_HIPBLAS_CHECK(hipblasSetPointerMode(*hipblas_pmd_handle_, HIPBLAS_POINTER_MODE_DEVICE));
  }
  return hipblas_pmd_handle_.get();
}

const hipblasHandle_t* RocmStreamHandle::hipblas_tensor_op_math_handle() {
  if (!hipblas_tensor_op_math_handle_) {
    hipblas_tensor_op_math_handle_.reset(new hipblasHandle_t);
    OF_HIPBLAS_CHECK(hipblasCreate(hipblas_tensor_op_math_handle_.get()));
    OF_HIPBLAS_CHECK(hipblasSetStream(*hipblas_tensor_op_math_handle_, *rocm_stream()));
// #if CUDA_VERSION >= 11000
//     OF_CUBLAS_CHECK(cublasSetMathMode(*hipblas_tensor_op_math_handle_, CUBLAS_DEFAULT_MATH));
// #else
    // OF_HIPBLAS_CHECK(cublasSetMathMode(*hipblas_tensor_op_math_handle_, CUBLAS_TENSOR_OP_MATH));
// #endif
  }
  return hipblas_tensor_op_math_handle_.get();
}

const miopenHandle_t* RocmStreamHandle::miopen_handle() {
  if (!miopen_handle_) {
    // if (IsCuda9OnTuringDevice()) {
      OF_ROCM_CHECK(hipDeviceSynchronize());
      OF_ROCM_CHECK(hipGetLastError());
    // }
    miopen_handle_.reset(new miopenHandle_t);
    OF_MIOPEN_CHECK(miopenCreate(miopen_handle_.get()));
    // if (IsCuda9OnTuringDevice()) {
      OF_ROCM_CHECK(hipDeviceSynchronize());
      hipGetLastError();
    // }
    OF_MIOPEN_CHECK(miopenSetStream(*miopen_handle_, *rocm_stream()));
  }
  return miopen_handle_.get();
}

void RocmStreamHandle::AddCallBack(std::function<void()> callback) {
  RocmCBEvent cb_event;
  cb_event.callback = std::move(callback);
  OF_ROCM_CHECK(hipEventCreateWithFlags(&(cb_event.event), hipEventDisableTiming));
  OF_ROCM_CHECK(hipEventRecord(cb_event.event, *rocm_stream()));
  cb_event_chan_->Send(cb_event);
}

RocmStreamHandle::~RocmStreamHandle() {
  if (rocm_stream_) { OF_ROCM_CHECK(hipStreamDestroy(*rocm_stream_)); }
}

#endif  // WITH_HIP

}  // namespace oneflow
