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
#include <gtest/gtest.h>
#include "oneflow/core/ep/test/primitive/primitive_test.h"
#include "oneflow/core/ep/include/primitive/memset.h"
#include "oneflow/core/ep/include/primitive/memcpy.h"
#include "oneflow/core/ep/include/primitive/fill.h"

#ifdef WITH_CUDA
#include <cuda.h>
#include <cuda_fp16.h>
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000
#endif  // WITH_CUDA

namespace oneflow {

namespace ep {

namespace primitive {

namespace test {

namespace {

template<DataType data_type, typename T>
void TestFill(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types, size_t n) {
  const size_t vector_size = n * sizeof(T);
  for (const auto& device_type : device_types) {
#ifdef WITH_CUDA
#if CUDA_VERSION >= 11000
    if (device_type == DeviceType::kCPU && data_type == DataType::kBFloat16) { continue; }
#endif  // CUDA_VERSION >= 11000
#endif  // WITH_CUDA
    auto device = registry->GetDevice(device_type, 0);
    ep::test::DeviceMemoryGuard device_mem(device.get(), vector_size);
    ep::test::PinnedMemoryGuard host_mem(device.get(), vector_size);
    ep::test::StreamGuard stream(device.get());

    std::unique_ptr<Fill> fill = NewPrimitive<FillFactory>(device_type, data_type);
    ASSERT_TRUE(fill.operator bool());
    std::unique_ptr<Memcpy> d2h = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kDtoH);
    ASSERT_TRUE(d2h.operator bool());

    fill->Launch(stream.stream(), device_mem.ptr(), Scalar(15.0), n);
    d2h->Launch(stream.stream(), host_mem.ptr(), device_mem.ptr(), vector_size);
    CHECK_JUST(stream.stream()->Sync());
    for (size_t i = 0; i < n; ++i) {
      ASSERT_EQ(*reinterpret_cast<T*>(host_mem.ptr<T>() + i), static_cast<T>(15.0));
    }

    fill->Launch(stream.stream(), device_mem.ptr(), Scalar(0), n);
    d2h->Launch(stream.stream(), host_mem.ptr(), device_mem.ptr(), vector_size);
    CHECK_JUST(stream.stream()->Sync());
    for (size_t i = 0; i < n; ++i) { ASSERT_EQ(*reinterpret_cast<T*>(host_mem.ptr<T>() + i), 0); }
  }
}

}  // namespace

TEST_F(PrimitiveTest, TestFill) {
  TestFill<DataType::kChar, char>(&device_manager_registry_, available_device_types_, 1024);
  TestFill<DataType::kDouble, double>(&device_manager_registry_, available_device_types_, 1024);
  TestFill<DataType::kFloat, float>(&device_manager_registry_, available_device_types_, 1024);
  TestFill<DataType::kInt8, int8_t>(&device_manager_registry_, available_device_types_, 1024);
  TestFill<DataType::kInt32, int32_t>(&device_manager_registry_, available_device_types_, 1024);
  TestFill<DataType::kInt64, int64_t>(&device_manager_registry_, available_device_types_, 1024);
  TestFill<DataType::kUInt8, uint8_t>(&device_manager_registry_, available_device_types_, 1024);
#ifdef WITH_CUDA
  TestFill<DataType::kFloat16, half>(&device_manager_registry_, available_device_types_, 1024);
#if CUDA_VERSION >= 11000
  TestFill<DataType::kBFloat16, nv_bfloat16>(&device_manager_registry_, available_device_types_,
                                             1024);
#endif  // CUDA_VERSION >= 11000
#endif  // WITH_CUDA
  TestFill<DataType::kBool, bool>(&device_manager_registry_, available_device_types_, 1024);
}

}  // namespace test

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow
