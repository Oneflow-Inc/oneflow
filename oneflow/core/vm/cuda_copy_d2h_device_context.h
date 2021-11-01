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
#ifndef ONEFLOW_CORE_DEVICE_CUDA_COPY_D2H_DEVICE_CONTEXT_H_
#define ONEFLOW_CORE_DEVICE_CUDA_COPY_D2H_DEVICE_CONTEXT_H_

#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/device/cuda_stream_handle.h"
#include "oneflow/core/device/cuda_event.h"
#include "oneflow/core/vm/cuda_host_allocator.h"

namespace oneflow {
namespace vm {

#ifdef WITH_CUDA

class CudaCopyD2HDeviceCtx : public DeviceCtx, public SingleThreadQueryCudaEventProvider {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaCopyD2HDeviceCtx);
  CudaCopyD2HDeviceCtx() = delete;
  ~CudaCopyD2HDeviceCtx() override = default;

  CudaCopyD2HDeviceCtx(int64_t device_id)
      : DeviceCtx(),
        SingleThreadQueryCudaEventProvider(device_id),
        cuda_handler_(new CudaStreamHandle(nullptr)),
        cuda_allocator_(std::make_unique<CudaHostAllocator>(device_id)),
        device_id_(device_id) {}

  cudaStream_t cuda_stream() const override { return cuda_handler_->cuda_stream(); }
  cublasHandle_t cublas_handle() const override { return cuda_handler_->cublas_handle(); }
  cudnnHandle_t cudnn_handle() const override { return cuda_handler_->cudnn_handle(); }

  void SyncDevice() override { OF_CUDA_CHECK(cudaStreamSynchronize(cuda_stream())); }

  void AddCallBack(std::function<void()> callback) const override { UNIMPLEMENTED(); }

  vm::Allocator* mut_allocator() override { return cuda_allocator_.get(); }

  DeviceType device_type() const override { return DeviceType::kGPU; }

 protected:
  std::unique_ptr<CudaStreamHandle> cuda_handler_;
  std::unique_ptr<CudaHostAllocator> cuda_allocator_;
  int64_t device_id_;
};

#endif  // WITH_CUDA
}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CUDA_COPY_D2H_DEVICE_CONTEXT_H_
