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
#include "oneflow/core/ep/include/primitive/memcpy.h"

namespace oneflow {

namespace ep {

namespace primitive {

namespace test {

TEST_F(PrimitiveTest, TestMemcpy) {
  const size_t test_elem = 1024 * 1024;
  const size_t test_size = test_elem * sizeof(float);
  for (const auto& device_type : available_device_types_) {
    auto device = device_manager_registry_.GetDevice(device_type, 0);
    ep::test::PinnedMemoryGuard input(device.get(), test_size);
    ep::test::PinnedMemoryGuard output(device.get(), test_size);
    ep::test::DeviceMemoryGuard device0(device.get(), test_size);
    ep::test::DeviceMemoryGuard device1(device.get(), test_size);
    for (size_t i = 0; i < test_elem; ++i) { *(input.ptr<float>() + i) = i; }
    ep::test::StreamGuard stream(device.get());
    std::unique_ptr<Memcpy> h2d = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kHtoD);
    ASSERT_TRUE(h2d.operator bool());
    std::unique_ptr<Memcpy> d2d = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kDtoD);
    ASSERT_TRUE(d2d.operator bool());
    std::unique_ptr<Memcpy> d2h = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kDtoH);
    ASSERT_TRUE(d2h.operator bool());
    h2d->Launch(stream.stream(), device0.ptr(), input.ptr(), test_size);
    d2d->Launch(stream.stream(), device1.ptr(), device0.ptr(), test_size);
    d2h->Launch(stream.stream(), output.ptr(), device1.ptr(), test_size);
    CHECK_JUST(stream.stream()->Sync());
    for (size_t i = 0; i < test_elem; ++i) {
      ASSERT_EQ(*(input.ptr<float>() + i), *(output.ptr<float>() + i));
    }
  }
}

}  // namespace test

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow
