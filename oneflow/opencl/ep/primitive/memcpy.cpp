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
#include "oneflow/core/ep/include/primitive/memcpy.h"

#include "oneflow/opencl/common/cl_util.h"
#include "oneflow/opencl/ep/cl_stream.h"

namespace oneflow {
namespace ep {
namespace primitive {

namespace {

class MemcpyImpl : public Memcpy {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemcpyImpl);
  MemcpyImpl(MemcpyKind kind) : kind_(kind) {}
  ~MemcpyImpl() override = default;

  void Launch(Stream* stream, void* dst, const void* src, size_t count) override {
    if (dst == src) { return; }
    auto* cl_stream = stream->As<clStream>();
    OF_CL_CHECK(clMemcpyAsync(dst, const_cast<void*>(src), count, kind_, cl_stream->cl_stream()));
    // Synchronous the stream since the host memory may not be page-locked, and clMemcpyAsync will
    // not translate to synchronous automatically like cuda.
    if (kind_ != MemcpyKind::kDtoD) { cl_stream->Sync(); }
  }

 private:
  MemcpyKind kind_;
};

class MemcpyFactoryImpl : public MemcpyFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemcpyFactoryImpl);
  MemcpyFactoryImpl() = default;
  ~MemcpyFactoryImpl() override = default;

  std::unique_ptr<Memcpy> New(MemcpyKind kind) override {
    return std::unique_ptr<Memcpy>(new MemcpyImpl(kind));
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kOpenCL, MemcpyFactory, MemcpyFactoryImpl);

}  // namespace

}  // namespace primitive
}  // namespace ep
}  // namespace oneflow
