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
#include "oneflow/core/ep/include/primitive/copy_nd.h"

namespace oneflow {

namespace ep {

namespace primitive {

namespace test {

namespace {

template<DataType data_type, typename T>
void TestCopyNd(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types,
                int64_t num_dims) {
  std::vector<int64_t> src_dims(num_dims, 0);
  std::vector<int64_t> src_pos(num_dims, 0);
  std::vector<int64_t> dst_pos(num_dims, 0);
  std::vector<int64_t> dst_dims(num_dims, 0);
  std::vector<int64_t> extent(num_dims, 0);
  int64_t src_elem = 1;
  int64_t dst_elem = 1;
  for (int i = 0; i < num_dims; ++i) {
    int64_t rand_dim = 8 + std::rand() % 32;
    int64_t rand_pos = std::rand() % 16;
    src_dims.at(i) = rand_dim;
    dst_pos.at(i) = rand_pos;
    dst_dims.at(i) = rand_pos + rand_dim;
    extent.at(i) = rand_dim;
    src_elem *= src_dims.at(i);
    dst_elem *= dst_dims.at(i);
  }
  int64_t src_size = src_elem * sizeof(T);
  int64_t dst_size = dst_elem * sizeof(T);

  for (const auto& device_type : device_types) {
    auto device = registry->GetDevice(device_type, 0);
    ep::test::PinnedMemoryGuard input(device.get(), src_size);
    ep::test::PinnedMemoryGuard output(device.get(), src_size);
    ep::test::DeviceMemoryGuard device0(device.get(), src_size);
    ep::test::DeviceMemoryGuard device1(device.get(), dst_size);
    for (size_t i = 0; i < src_elem; ++i) { *(input.ptr<T>() + i) = static_cast<T>(i); }
    ep::test::StreamGuard stream(device.get());
    std::unique_ptr<Memcpy> h2d = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kHtoD);
    ASSERT_TRUE(h2d.operator bool());
    std::unique_ptr<CopyNd> copy_nd = NewPrimitive<CopyNdFactory>(device_type, num_dims);
    ASSERT_TRUE(copy_nd.operator bool());
    std::unique_ptr<Memcpy> d2h = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kDtoH);
    ASSERT_TRUE(d2h.operator bool());
    std::unique_ptr<Memset> memset = NewPrimitive<MemsetFactory>(device_type);
    ASSERT_TRUE(memset.operator bool());
    h2d->Launch(stream.stream(), device0.ptr(), input.ptr(), src_size);
    // contiguous device0 to noncontiguous device1
    copy_nd->Launch(stream.stream(), data_type, num_dims, device1.ptr(), dst_dims.data(),
                    dst_pos.data(), device0.ptr(), src_dims.data(), src_pos.data(), extent.data());
    // memset device0
    memset->Launch(stream.stream(), device0.ptr(), 0x55, src_size);
    // noncontiguous device1 to contiguous device0
    copy_nd->Launch(stream.stream(), data_type, num_dims, device0.ptr(), src_dims.data(),
                    src_pos.data(), device1.ptr(), dst_dims.data(), dst_pos.data(), extent.data());
    d2h->Launch(stream.stream(), output.ptr(), device0.ptr(), src_size);
    CHECK_JUST(stream.stream()->Sync());
    for (size_t i = 0; i < src_elem; ++i) {
      ASSERT_EQ(*(input.ptr<T>() + i), *(output.ptr<T>() + i));
    }
  }
}

}  // namespace

TEST_F(PrimitiveTest, TestCopyNd) {
  for (int i = 1; i < 6; ++i) {
    TestCopyNd<DataType::kDouble, double>(&device_manager_registry_, available_device_types_, i);
    TestCopyNd<DataType::kFloat, float>(&device_manager_registry_, available_device_types_, i);
    TestCopyNd<DataType::kInt8, int8_t>(&device_manager_registry_, available_device_types_, i);
    TestCopyNd<DataType::kInt32, int32_t>(&device_manager_registry_, available_device_types_, i);
    TestCopyNd<DataType::kInt64, int64_t>(&device_manager_registry_, available_device_types_, i);
  }
}

}  // namespace test

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow
