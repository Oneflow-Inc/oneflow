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
#include <unsupported/Eigen/CXX11/Tensor>

namespace oneflow {

namespace ep {

namespace primitive {

namespace test {

namespace {

template<BinaryOp binary_op, DataType src_data_type, typename Src, DataType dst_data_type,
         typename Dst, size_t m, size_t n>
void TestElementwiseBinary(DeviceManagerRegistry* registry,
                           const std::set<DeviceType>& device_types) {
  using SrcMatrix = Eigen::Matrix<Src, m, n>;
  using DstMatrix = Eigen::Matrix<Dst, m, n>;
  SrcMatrix a = SrcMatrix::Random();
  SrcMatrix b = SrcMatrix::Random() + SrcMatrix::Constant(static_cast<Src>(1));
  DstMatrix c;
  std::cout << "a" << a << std::endl;
  std::cout << "b" << b << std::endl;
  if (binary_op == BinaryOp::kAdd) {
    c = (a + b).template cast<Dst>();
  } else if (binary_op == BinaryOp::kSub) {
    c = (a - b).template cast<Dst>();
  } else if (binary_op == BinaryOp::kMul) {
    c = (a.array() * b.array()).template cast<Dst>();
  } else if (binary_op == BinaryOp::kDiv) {
    c = (a.array() / b.array()).template cast<Dst>();
  } else if (binary_op == BinaryOp::kMax) {
    c = (a.array().max(b.array())).template cast<Dst>();
  } else if (binary_op == BinaryOp::kMin) {
    c = (a.array().min(b.array())).template cast<Dst>();
  } else if (binary_op == BinaryOp::kPow) {
    c = (a.array().pow(b.array())).template cast<Dst>();  // nan
  } else if (binary_op == BinaryOp::kEqual) {
    c = (a.array() == b.array()).template cast<Dst>();
  } else if (binary_op == BinaryOp::kNotEqual) {
    c = (a.array() != b.array()).template cast<Dst>();
  } else if (binary_op == BinaryOp::kLessThan) {
    c = (a.array() < b.array()).template cast<Dst>();
  } else if (binary_op == BinaryOp::kLessEqual) {
    c = (a.array() <= b.array()).template cast<Dst>();
  } else if (binary_op == BinaryOp::kGreaterThan) {
    c = (a.array() > b.array()).template cast<Dst>();
  } else if (binary_op == BinaryOp::kGreaterEqual) {
    c = (a.array() >= b.array()).template cast<Dst>();
  } else if (binary_op == BinaryOp::kLogicalAnd) {
    c = (a.array().template cast<bool>() && b.array().template cast<bool>()).template cast<Dst>();
  } else if (binary_op == BinaryOp::kLogicalOr) {
    c = (a.array().template cast<bool>() || b.array().template cast<bool>()).template cast<Dst>();
  } else if (binary_op == BinaryOp::kLogicalXor) {
    c = (a.array().template cast<bool>() ^ b.array().template cast<bool>()).template cast<Dst>();
  }
  int64_t num_a_dims = 2;
  std::vector<int64_t> a_dims = {m, n};
  int64_t num_b_dims = 2;
  std::vector<int64_t> b_dims = {m, n};
  int64_t a_size = m * n * sizeof(Src);
  int64_t b_size = m * n * sizeof(Src);
  int64_t c_size = m * n * sizeof(Dst);
  int num_axes = 2;

  for (const auto& device_type : device_types) {
    LOG(ERROR) << "device " << device_type << " dtype " << src_data_type << " binary " << binary_op;
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
        NewPrimitive<BroadcastElementwiseBinaryFactory>(device_type, binary_op, src_data_type,
                                                        dst_data_type, num_axes);
    ASSERT_TRUE(d2h.operator bool());
    ASSERT_TRUE(h2d.operator bool());
    ASSERT_TRUE(binary.operator bool());
    h2d->Launch(stream.stream(), device_a.ptr(), input_a.ptr(), a_size);
    h2d->Launch(stream.stream(), device_b.ptr(), input_b.ptr(), b_size);
    binary->Launch(stream.stream(), num_a_dims, a_dims.data(), device_a.ptr(), num_b_dims,
                   b_dims.data(), device_b.ptr(), device_c.ptr());
    d2h->Launch(stream.stream(), output.ptr(), device_c.ptr(), c_size);
    CHECK_JUST(stream.stream()->Sync());
    auto res = Eigen::Map<DstMatrix, Eigen::Unaligned>(reinterpret_cast<Dst*>(output.ptr()), m, n);
    ASSERT_TRUE(c.template isApprox(res));
  }
}

template<BinaryOp binary_op, size_t m, size_t n>
void TestComputeBinary(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types) {
  TestElementwiseBinary<binary_op, DataType::kDouble, double, DataType::kDouble, double, 16, 8>(
      registry, device_types);
  TestElementwiseBinary<binary_op, DataType::kFloat, float, DataType::kFloat, float, 16, 8>(
      registry, device_types);
  TestElementwiseBinary<binary_op, DataType::kFloat16, Eigen::half, DataType::kFloat16, Eigen::half,
                        16, 8>(registry, device_types);
}

template<BinaryOp binary_op, size_t m, size_t n>
void TestLogicalBinary(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types) {
  TestElementwiseBinary<binary_op, DataType::kDouble, double, DataType::kBool, bool, 16, 8>(
      registry, device_types);
  TestElementwiseBinary<binary_op, DataType::kFloat, float, DataType::kBool, bool, 16, 8>(
      registry, device_types);
  TestElementwiseBinary<binary_op, DataType::kFloat16, Eigen::half, DataType::kBool, bool, 16, 8>(
      registry, device_types);
}

}  // namespace

TEST_F(PrimitiveTest, TestBinary) {
  TestComputeBinary<BinaryOp::kAdd, 16, 8>(&device_manager_registry_, available_device_types_);
  TestComputeBinary<BinaryOp::kSub, 16, 8>(&device_manager_registry_, available_device_types_);
  TestComputeBinary<BinaryOp::kMul, 16, 8>(&device_manager_registry_, available_device_types_);
  TestComputeBinary<BinaryOp::kDiv, 16, 8>(&device_manager_registry_, available_device_types_);
  TestComputeBinary<BinaryOp::kMax, 16, 8>(&device_manager_registry_, available_device_types_);
  TestComputeBinary<BinaryOp::kMin, 16, 8>(&device_manager_registry_, available_device_types_);
  TestLogicalBinary<BinaryOp::kEqual, 16, 8>(&device_manager_registry_, available_device_types_);
  TestLogicalBinary<BinaryOp::kNotEqual, 16, 8>(&device_manager_registry_, available_device_types_);
  TestLogicalBinary<BinaryOp::kLessThan, 16, 8>(&device_manager_registry_, available_device_types_);
  TestLogicalBinary<BinaryOp::kLessEqual, 16, 8>(&device_manager_registry_,
                                                 available_device_types_);
  TestLogicalBinary<BinaryOp::kGreaterThan, 16, 8>(&device_manager_registry_,
                                                   available_device_types_);
  TestLogicalBinary<BinaryOp::kGreaterEqual, 16, 8>(&device_manager_registry_,
                                                    available_device_types_);
  TestLogicalBinary<BinaryOp::kLogicalAnd, 16, 8>(&device_manager_registry_,
                                                  available_device_types_);
  TestLogicalBinary<BinaryOp::kLogicalOr, 16, 8>(&device_manager_registry_,
                                                 available_device_types_);
  TestLogicalBinary<BinaryOp::kLogicalXor, 16, 8>(&device_manager_registry_,
                                                  available_device_types_);
}

}  // namespace test

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow
