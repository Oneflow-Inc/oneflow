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
#include "oneflow/core/device/device_context_adapter.h"
#include "oneflow/core/vm/cpu_allocator.h"
#include "oneflow/core/device/cuda_event_record.h"

#ifdef WITH_CUDA

#include "oneflow/core/ep/cuda/cuda_stream.h"

#endif  // WITH_CUDA

namespace oneflow {

namespace {

class CpuDeviceCtxAdapter final : public DeviceCtx, public EventRecordProvider {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuDeviceCtxAdapter);
  explicit CpuDeviceCtxAdapter(ep::Stream* stream) : stream_(stream) {}
  ~CpuDeviceCtxAdapter() override = default;

  std::unique_ptr<DeviceCtx> Copy() const {
    return std::unique_ptr<DeviceCtx>(new CpuDeviceCtxAdapter(stream_));
  }

  ep::Stream* stream() override { return stream_; }

  vm::Allocator* mut_allocator() override { return Global<vm::CpuAllocator>::Get(); }

  DeviceType device_type() const override { return stream_->device_type(); }

  std::shared_ptr<EventRecord> MakeEventRecord() override {
    return std::make_shared<NaiveEventRecord>();
  }

 private:
  ep::Stream* stream_;
};

#ifdef WITH_CUDA

class CudaDeviceCtxAdapter : public DeviceCtx, public EventRecordProvider {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaDeviceCtxAdapter);
  explicit CudaDeviceCtxAdapter(ep::Stream* stream) {
    cuda_stream_ = CHECK_NOTNULL(dynamic_cast<ep::CudaStream*>(stream));
  }
  ~CudaDeviceCtxAdapter() override = default;

  cudaStream_t cuda_stream() const override { return cuda_stream_->cuda_stream(); }
  cublasHandle_t cublas_handle() const override { return cuda_stream_->cublas_handle(); }
  cudnnHandle_t cudnn_handle() const override { return cuda_stream_->cudnn_handle(); }

  DeviceType device_type() const override { return cuda_stream_->device_type(); }

  ep::Stream* stream() override { return cuda_stream_; }

  std::shared_ptr<EventRecord> MakeEventRecord() override {
    return std::make_shared<CudaEventRecord>(this);
  }

 protected:
  ep::CudaStream* cuda_stream_;
};

#endif  // WITH_CUDA

}  // namespace

DeviceCtx* NewDeviceCtxAdapter(ep::Stream* stream) {
  if (stream->device_type() == DeviceType::kCPU) {
    return new CpuDeviceCtxAdapter(stream);
  } else if (stream->device_type() == DeviceType::kCUDA) {
#ifdef WITH_CUDA
    return new CudaDeviceCtxAdapter(stream);
#else
    UNIMPLEMENTED();
    return nullptr;
#endif  // WITH_CUDA
  } else {
    UNIMPLEMENTED();
    return nullptr;
  }
}

}  // namespace oneflow
