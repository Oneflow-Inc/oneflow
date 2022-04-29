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
#include "oneflow/core/ep/include/primitive/add.h"
#include <Eigen/Core>

namespace oneflow {

namespace ep {

namespace primitive {

namespace test {

namespace {

template<DataType data_type, typename T, size_t n>
void TestAdd(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types) {
  constexpr size_t max_arity = 10;
  using Matrix = Eigen::Matrix<T, 1, n>;
  std::vector<Matrix> srcs(max_arity);
  std::vector<Matrix> dsts(max_arity);
  for (size_t i = 0; i < max_arity; ++i) {
    srcs[i] = Matrix::Random();
    if (i == 0) {
      dsts[i] = Matrix::Zero();
    } else {
      dsts[i] = srcs[i - 1] + dsts[i - 1];
    }
  }
  const size_t vector_size = n * sizeof(T);
  for (const auto& device_type : device_types) {
    auto device = registry->GetDevice(device_type, 0);
    std::vector<void*> host_srcs(max_arity);
    std::vector<void*> device_srcs(max_arity);
    std::vector<void*> host_dsts(max_arity);
    std::vector<void*> device_dsts(max_arity);
    AllocationOptions pinned_options;
    pinned_options.SetPinnedDevice(device_type, 0);
    AllocationOptions device_options;
    for (size_t i = 0; i < max_arity; ++i) {
      CHECK_JUST(device->AllocPinned(pinned_options, &host_srcs[i], vector_size));
      CHECK_JUST(device->AllocPinned(pinned_options, &host_dsts[i], vector_size));
      CHECK_JUST(device->Alloc(device_options, &device_srcs[i], vector_size));
      CHECK_JUST(device->Alloc(device_options, &device_dsts[i], vector_size));
    }
    ep::test::StreamGuard stream(device.get());
    std::unique_ptr<Add> add = NewPrimitive<AddFactory>(device_type, data_type);
    ASSERT_TRUE(add.operator bool());
    std::unique_ptr<Memcpy> h2d = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kHtoD);
    std::unique_ptr<Memcpy> d2h = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kDtoH);
    ASSERT_TRUE(d2h.operator bool());
    ASSERT_TRUE(h2d.operator bool());
    for (size_t i = 0; i < max_arity; ++i) {
      std::memcpy(host_srcs[i], srcs[i].data(), vector_size);
      h2d->Launch(stream.stream(), device_srcs[i], host_srcs[i], vector_size);
    }
    for (size_t i = 2; i < max_arity; ++i) {
      add->Launch(stream.stream(), device_srcs.data(), i, device_dsts.at(i), n);
    }
    for (size_t i = 2; i < max_arity; ++i) {
      d2h->Launch(stream.stream(), host_dsts[i], device_dsts[i], vector_size);
    }
    CHECK_JUST(stream.stream()->Sync());
    for (size_t i = 2; i < max_arity; ++i) {
      auto res = Eigen::Map<Matrix, Eigen::Unaligned>(reinterpret_cast<T*>(host_dsts[i]), n);
      ASSERT_TRUE(dsts[i].template isApprox(res));
    }
    for (size_t i = 0; i < max_arity; ++i) {
      device->FreePinned(pinned_options, host_srcs[i]);
      device->FreePinned(pinned_options, host_dsts[i]);
      device->Free(device_options, device_srcs[i]);
      device->Free(device_options, device_dsts[i]);
    }
  }
}

}  // namespace

TEST_F(PrimitiveTest, TestAdd) {
  TestAdd<DataType::kDouble, double, 1024>(&device_manager_registry_, available_device_types_);
  TestAdd<DataType::kFloat, float, 1024>(&device_manager_registry_, available_device_types_);
  TestAdd<DataType::kFloat16, Eigen::half, 1024>(&device_manager_registry_,
                                                 available_device_types_);
}

}  // namespace test

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow
