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
#ifdef WITH_CUDA

#include "oneflow/core/primitive/memcpy.h"
#include "oneflow/core/stream/cuda_stream_context.h"
#include <cuda_runtime.h>

namespace oneflow {

namespace primitive {

namespace {

class MemcpyImpl : public Memcpy {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemcpyImpl);
  MemcpyImpl() = default;
  ~MemcpyImpl() override = default;

  void Launch(StreamContext* stream_ctx, void* dst, const void* src, size_t count) override {
    auto* cuda_stream_ctx = CHECK_NOTNULL(dynamic_cast<CudaStreamContext*>(stream_ctx));
    OF_CUDA_CHECK(
        cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, cuda_stream_ctx->cuda_stream()));
  }
};

class MemcpyFactoryImpl : public MemcpyFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemcpyFactoryImpl);
  MemcpyFactoryImpl() = default;
  ~MemcpyFactoryImpl() override = default;

  std::unique_ptr<Memcpy> New(MemcpyKind kind) override {
    return std::unique_ptr<Memcpy>(new MemcpyImpl());
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kGPU, MemcpyFactory, MemcpyFactoryImpl);

}  // namespace

}  // namespace primitive

}  // namespace oneflow

#endif
