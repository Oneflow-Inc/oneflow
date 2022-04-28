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
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"
#include <Eigen/Core>

namespace oneflow {

namespace ep {

namespace primitive {

namespace test {

namespace {

template<BinaryOp binary_op, DataType data_type, typename T, size_t m, size_t n>
void TestElementwiseBinary(DeviceManagerRegistry* registry,
                           const std::set<DeviceType>& device_types) {
  using Matrix = Eigen::Matrix<T, m, n>;
  Matrix a = Matrix::Random();
  Matrix b = Matrix::Random();
  Matrix c = Matrix::Zero();
  c = a + b;
  int64_t num_a_dims = 2;
  std::vector<int64_t> a_dims = {m, n};
  int64_t num_b_dims = 2;
  std::vector<int64_t> b_dims = {m, n};
  int64_t a_size = m * n * sizeof(T);
  int64_t b_size = m * n * sizeof(T);
  int64_t c_size = m * n * sizeof(T);
  int num_axes = 2;

  for (const auto& device_type : device_types) {
    // TODO: onednn C++ exception with description "could not create a primitive descriptor
    // iterator" thrown in the test body.
    if (device_type != DeviceType::kCPU && data_type == DataType::kFloat16) { continue; }
    auto device = registry->GetDevice(device_type, 0);
    ep::test::PinnedMemoryGuard input_a(device.get(), a_size);
    ep::test::PinnedMemoryGuard input_b(device.get(), b_size);
    std::memcpy(input_a.ptr(), a.data(), a_size);
    std::memcpy(input_b.ptr(), b.data(), b_size);

    ep::test::PinnedMemoryGuard output(device.get(), c_size);
    ep::test::DeviceMemoryGuard device_a(device.get(), a_size);
    ep::test::DeviceMemoryGuard device_b(device.get(), b_size);
    ep::test::DeviceMemoryGuard device_c(device.get(), c_size);
    ep::test::StreamGuard stream(device.get());
    std::unique_ptr<Memcpy> h2d = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kHtoD);
    std::unique_ptr<Memcpy> d2h = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kDtoH);
    std::unique_ptr<BroadcastElementwiseBinary> binary =
        NewPrimitive<BroadcastElementwiseBinaryFactory>(device_type, binary_op, data_type,
                                                        data_type, num_axes);
    ASSERT_TRUE(d2h.operator bool());
    ASSERT_TRUE(h2d.operator bool());
    ASSERT_TRUE(binary.operator bool());
    h2d->Launch(stream.stream(), device_a.ptr(), input_a.ptr(), a_size);
    h2d->Launch(stream.stream(), device_b.ptr(), input_b.ptr(), b_size);
    binary->Launch(stream.stream(), num_a_dims, a_dims.data(), device_a.ptr(), num_b_dims,
                   b_dims.data(), device_b.ptr(), device_c.ptr());
    d2h->Launch(stream.stream(), output.ptr(), device_c.ptr(), c_size);
    CHECK_JUST(stream.stream()->Sync());
    auto res = Eigen::Map<Matrix, Eigen::Unaligned>(reinterpret_cast<T*>(output.ptr()), m, n);
    ASSERT_TRUE(c.template isApprox(res));
  }
}

}  // namespace

TEST_F(PrimitiveTest, TestBinary) {
  TestElementwiseBinary<BinaryOp::kAdd, DataType::kDouble, double, 16, 16>(
      &device_manager_registry_, available_device_types_);
  TestElementwiseBinary<BinaryOp::kAdd, DataType::kFloat, float, 16, 16>(&device_manager_registry_,
                                                                         available_device_types_);
  TestElementwiseBinary<BinaryOp::kAdd, DataType::kFloat16, Eigen::half, 16, 16>(
      &device_manager_registry_, available_device_types_);
}

}  // namespace test

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow
