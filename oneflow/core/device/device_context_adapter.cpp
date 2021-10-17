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
#include "oneflow/core/stream/cuda_stream_context.h"
#include "oneflow/core/vm/cpu_allocator.h"
#include "oneflow/core/device/cuda_event_record.h"

namespace oneflow {

namespace {

class CpuDeviceCtxAdapter final : public DeviceCtx, public EventRecordProvider {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuDeviceCtxAdapter);
  explicit CpuDeviceCtxAdapter(StreamContext* stream_ctx) : stream_ctx_(stream_ctx) {}
  ~CpuDeviceCtxAdapter() override = default;

  std::unique_ptr<DeviceCtx> Copy() const {
    return std::unique_ptr<DeviceCtx>(new CpuDeviceCtxAdapter(stream_ctx_));
  }

  void SyncDevice() override {}
  void AddCallBack(std::function<void()> callback) const override { callback(); }

  vm::Allocator* mut_allocator() override { return Global<vm::CpuAllocator>::Get(); }

  DeviceType device_type() const override { return stream_ctx_->device_type(); }

  std::shared_ptr<EventRecord> MakeEventRecord() override {
    return std::make_shared<NaiveEventRecord>();
  }

 private:
  StreamContext* stream_ctx_;
};

#ifdef WITH_CUDA

class CudaDeviceCtxAdapter : public DeviceCtx, public EventRecordProvider {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaDeviceCtxAdapter);
  explicit CudaDeviceCtxAdapter(CudaStreamContext* stream_ctx) : stream_ctx_(stream_ctx) {}
  ~CudaDeviceCtxAdapter() override = default;

  cudaStream_t cuda_stream() const override { return stream_ctx_->cuda_stream(); }
  cublasHandle_t cublas_handle() const override { return stream_ctx_->cublas_handle(); }
  cudnnHandle_t cudnn_handle() const override { return stream_ctx_->cudnn_handle(); }

  void SyncDevice() override { CHECK_JUST(stream_ctx_->Sync()); }

  void AddCallBack(std::function<void()> callback) const override {
    CHECK_JUST(stream_ctx_->AddCallback(std::move(callback)));
  }

  DeviceType device_type() const override { return stream_ctx_->device_type(); }

  std::shared_ptr<EventRecord> MakeEventRecord() override {
    return std::make_shared<CudaEventRecord>(this);
  }

 protected:
  CudaStreamContext* stream_ctx_;
};

#endif  // WITH_CUDA

}  // namespace

DeviceCtx* NewDeviceCtxAdapter(StreamContext* ctx) {
  if (ctx->device_type() == DeviceType::kCPU) {
    return new CpuDeviceCtxAdapter(ctx);
  } else if (ctx->device_type() == DeviceType::kGPU) {
#ifdef WITH_CUDA
    return new CudaDeviceCtxAdapter(CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(ctx)));
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
