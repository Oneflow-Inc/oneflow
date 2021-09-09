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
#include "oneflow/core/stream/stream_context_adapter.h"
#include "oneflow/core/stream/cuda_stream_context.h"

namespace oneflow {

namespace {

class DeviceCtxStreamContextAdapter : public StreamContext {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceCtxStreamContextAdapter);
  explicit DeviceCtxStreamContextAdapter(DeviceCtx* device_ctx) : device_ctx_(device_ctx) {}
  ~DeviceCtxStreamContextAdapter() override = default;

  Maybe<void> AddCallback(std::function<void()> callback) override {
    device_ctx_->AddCallBack(std::move(callback));
    return Maybe<void>::Ok();
  }

  Maybe<void> Sync() override {
    device_ctx_->SyncDevice();
    return Maybe<void>::Ok();
  }

  DeviceType device_type() const override { return device_ctx_->device_type(); }

 private:
  DeviceCtx* device_ctx_;
};

#ifdef WITH_CUDA

class CudaDeviceCtxStreamContextAdapter : public CudaStreamContext {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaDeviceCtxStreamContextAdapter);
  explicit CudaDeviceCtxStreamContextAdapter(DeviceCtx* device_ctx) : device_ctx_(device_ctx) {}
  ~CudaDeviceCtxStreamContextAdapter() override = default;

  Maybe<void> AddCallback(std::function<void()> callback) override {
    device_ctx_->AddCallBack(std::move(callback));
    return Maybe<void>::Ok();
  }

  Maybe<void> Sync() override {
    device_ctx_->SyncDevice();
    return Maybe<void>::Ok();
  }

  DeviceType device_type() const override { return device_ctx_->device_type(); }

  cudaStream_t cuda_stream() const override { return device_ctx_->cuda_stream(); }

  cublasHandle_t cublas_pmh_handle() const override { return device_ctx_->cublas_pmh_handle(); }

  cublasHandle_t cublas_tensor_op_math_handle() const override {
    return device_ctx_->cublas_tensor_op_math_handle();
  }

  cublasHandle_t cublas_pmd_handle() const override { return device_ctx_->cublas_pmd_handle(); }

  cudnnHandle_t cudnn_handle() const override { return device_ctx_->cudnn_handle(); }

 private:
  DeviceCtx* device_ctx_;
};
#endif  // WITH_CUDA

}  // namespace

StreamContext* NewStreamContextAdapter(DeviceCtx* ctx) {
  if (ctx->device_type() == DeviceType::kGPU) {
#ifdef WITH_CUDA
    return new CudaDeviceCtxStreamContextAdapter(ctx);
#else
    UNIMPLEMENTED();
    return nullptr;
#endif  // WITH_CUDA
  } else {
    return new DeviceCtxStreamContextAdapter(ctx);
  }
}

}  // namespace oneflow
