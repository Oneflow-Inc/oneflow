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

namespace oneflow {

namespace ep {
namespace primitive {

namespace {

class MemcpyImpl : public Memcpy {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemcpyImpl);
  MemcpyImpl() = default;
  ~MemcpyImpl() = default;

  void Launch(Stream* stream, void* dst, const void* src, size_t count) {
    if (dst == src) { return; }
    std::memcpy(dst, src, count);
  }
};

class MemcpyFactoryImpl : public MemcpyFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemcpyFactoryImpl);
  MemcpyFactoryImpl() = default;
  ~MemcpyFactoryImpl() override = default;

  std::unique_ptr<Memcpy> New(MemcpyKind kind) override { return std::make_unique<MemcpyImpl>(); }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCPU, MemcpyFactory, MemcpyFactoryImpl);

}  // namespace

}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
