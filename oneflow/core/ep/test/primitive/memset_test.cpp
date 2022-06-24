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

namespace oneflow {

namespace ep {

namespace primitive {

namespace test {

TEST_F(PrimitiveTest, TestMemset) {
  const size_t test_size = 1024 * 1024;
  for (const auto& device_type : available_device_types_) {
    auto device = device_manager_registry_.GetDevice(device_type, 0);
    ep::test::DeviceMemoryGuard device_mem(device.get(), test_size);
    ep::test::PinnedMemoryGuard host_mem(device.get(), test_size);
    ep::test::StreamGuard stream(device.get());
    std::unique_ptr<Memset> memset = NewPrimitive<MemsetFactory>(device_type);
    ASSERT_TRUE(memset.operator bool());
    std::unique_ptr<Memcpy> d2h = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kDtoH);
    ASSERT_TRUE(d2h.operator bool());
    memset->Launch(stream.stream(), device_mem.ptr(), 0x55, test_size);
    d2h->Launch(stream.stream(), host_mem.ptr(), device_mem.ptr(), test_size);
    CHECK_JUST(stream.stream()->Sync());
    for (size_t i = 0; i < test_size; ++i) { ASSERT_EQ(*(host_mem.ptr<char>() + i), 0x55); }
    memset->Launch(stream.stream(), device_mem.ptr(), 0, test_size);
    d2h->Launch(stream.stream(), host_mem.ptr(), device_mem.ptr(), test_size);
    CHECK_JUST(stream.stream()->Sync());
    for (size_t i = 0; i < test_size; ++i) { ASSERT_EQ(*(host_mem.ptr<char>() + i), 0); }
  }
}

}  // namespace test

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow
