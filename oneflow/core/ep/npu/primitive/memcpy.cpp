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
#ifdef WITH_NPU

#include "oneflow/core/ep/include/primitive/memcpy.h"
#include "oneflow/core/ep/npu/npu_stream.h"

#include <iostream>
namespace oneflow {

namespace ep {
namespace primitive {

namespace {

class MemcpyImpl : public Memcpy {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemcpyImpl);
  MemcpyImpl() = delete;
  MemcpyImpl(MemcpyKind kind) : kind(kind){}
  ~MemcpyImpl() override = default;

  void Launch(Stream* stream, void* dst, const void* src, size_t count) override {
    if (dst == src) { return; }
    auto* npu_stream = stream->As<NpuStream>();
    auto aclRule = ACL_MEMCPY_HOST_TO_DEVICE;
    if(kind == MemcpyKind::kDtoH)
    {
      aclRule = ACL_MEMCPY_DEVICE_TO_HOST;
    } 
    else if(kind == MemcpyKind::kDtoD)
    {
      aclRule = ACL_MEMCPY_DEVICE_TO_DEVICE;
    }
    OF_NPU_CHECK(aclrtMemcpy(dst, 
                                  count, 
                                  src, 
                                  count, 
                                  aclRule));
  }
  MemcpyKind kind;
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

REGISTER_PRIMITIVE_FACTORY(DeviceType::kNPU, MemcpyFactory, MemcpyFactoryImpl);

}  // namespace

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow

#endif
