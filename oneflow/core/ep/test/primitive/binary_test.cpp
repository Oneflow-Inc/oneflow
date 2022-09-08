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

template<typename T>
Scalar GetScalar(const T& value) {
  return Scalar(value);
}

template<>
Scalar GetScalar<Eigen::half>(const Eigen::half& value) {
  return Scalar(static_cast<float>(value));
}

template<BinaryOp binary_op, DataType src_data_type, typename Src, DataType dst_data_type,
         typename Dst>
void TestElementwiseBroadcastBinary(DeviceManagerRegistry* registry,
                                    const std::set<DeviceType>& device_types, int test_type) {
  const int num_axes = 4;
  const int broadcast_dim0 = 16;
  const int broadcast_dim1 = 3;
  const int broadcast_dim2 = 4;
  const int broadcast_dim3 = 8;
  bool is_broadcast = false;
  bool left_scalar = false;
  bool right_scalar = false;
  if (test_type == 0) {
    // do nothing
  } else if (test_type == 1) {
    is_broadcast = true;
  } else if (test_type == 2) {
    left_scalar = true;
  } else if (test_type == 3) {
    right_scalar = true;
  } else {
    UNIMPLEMENTED();
  }
  const int a_dim0 = left_scalar ? 1 : broadcast_dim0;
  const int a_dim1 = left_scalar ? 1 : broadcast_dim1;
  const int a_dim2 = left_scalar ? 1 : broadcast_dim2;
  const int a_dim3 = left_scalar ? 1 : (is_broadcast ? 1 : broadcast_dim3);
  const int b_dim0 = right_scalar ? 1 : broadcast_dim0;
  const int b_dim1 = right_scalar ? 1 : (is_broadcast ? 1 : broadcast_dim1);
  const int b_dim2 = right_scalar ? 1 : broadcast_dim2;
  const int b_dim3 = right_scalar ? 1 : broadcast_dim3;
  const int a_broadcast0 = left_scalar ? broadcast_dim0 : 1;
  const int a_broadcast1 = left_scalar ? broadcast_dim1 : 1;
  const int a_broadcast2 = left_scalar ? broadcast_dim2 : 1;
  const int a_broadcast3 = left_scalar ? broadcast_dim3 : (is_broadcast ? broadcast_dim3 : 1);
  const int b_broadcast0 = right_scalar ? broadcast_dim0 : 1;
  const int b_broadcast1 = right_scalar ? broadcast_dim1 : (is_broadcast ? broadcast_dim1 : 1);
  const int b_broadcast2 = right_scalar ? broadcast_dim2 : 1;
  const int b_broadcast3 = right_scalar ? broadcast_dim3 : 1;
  const Eigen::array<int, 4> a_broadcast = {a_broadcast0, a_broadcast1, a_broadcast2, a_broadcast3};
  const Eigen::array<int, 4> b_broadcast = {b_broadcast0, b_broadcast1, b_broadcast2, b_broadcast3};
  Eigen::Tensor<Src, 4, Eigen::RowMajor> a(a_dim0, a_dim1, a_dim2, a_dim3);
  Eigen::Tensor<Src, 4, Eigen::RowMajor> b(b_dim0, b_dim1, b_dim2, b_dim3);
  Eigen::Tensor<Dst, 4, Eigen::RowMajor> c(broadcast_dim0, broadcast_dim1, broadcast_dim2,
                                           broadcast_dim3);
  a.setRandom();
  b.setRandom();
  if (binary_op == BinaryOp::kAdd) {
    c = (a.broadcast(a_broadcast) + b.broadcast(b_broadcast)).template cast<Dst>();
  } else if (binary_op == BinaryOp::kSub) {
    c = (a.broadcast(a_broadcast) - b.broadcast(b_broadcast)).template cast<Dst>();
  } else if (binary_op == BinaryOp::kMul) {
    c = (a.broadcast(a_broadcast) * b.broadcast(b_broadcast)).template cast<Dst>();
  } else if (binary_op == BinaryOp::kDiv) {
    Eigen::Tensor<Src, 4, Eigen::RowMajor> constant_value(b_dim0, b_dim1, b_dim2, b_dim3);
    // avoid div 0
    if (src_data_type == kInt8 || src_data_type == kUInt8) {
      int rand_value = std::rand() % 127;
      constant_value.setConstant(static_cast<Src>(rand_value));
      b = constant_value;
    } else {
      constant_value.setConstant(static_cast<Src>(1));
      b += constant_value;
    }
    c = (a.broadcast(a_broadcast) / b.broadcast(b_broadcast)).template cast<Dst>();
  } else if (binary_op == BinaryOp::kMax) {
    c = (a.broadcast(a_broadcast).cwiseMax(b.broadcast(b_broadcast))).template cast<Dst>();
  } else if (binary_op == BinaryOp::kMin) {
    c = (a.broadcast(a_broadcast).cwiseMin(b.broadcast(b_broadcast))).template cast<Dst>();
  } else if (binary_op == BinaryOp::kEqual) {
    c = (a.broadcast(a_broadcast) == b.broadcast(b_broadcast)).template cast<Dst>();
  } else if (binary_op == BinaryOp::kNotEqual) {
    c = (a.broadcast(a_broadcast) != b.broadcast(b_broadcast)).template cast<Dst>();
  } else if (binary_op == BinaryOp::kLessThan) {
    c = (a.broadcast(a_broadcast) < b.broadcast(b_broadcast)).template cast<Dst>();
  } else if (binary_op == BinaryOp::kLessEqual) {
    c = (a.broadcast(a_broadcast) <= b.broadcast(b_broadcast)).template cast<Dst>();
  } else if (binary_op == BinaryOp::kGreaterThan) {
    c = (a.broadcast(a_broadcast) > b.broadcast(b_broadcast)).template cast<Dst>();
  } else if (binary_op == BinaryOp::kGreaterEqual) {
    c = (a.broadcast(a_broadcast) >= b.broadcast(b_broadcast)).template cast<Dst>();
  } else if (binary_op == BinaryOp::kLogicalAnd) {
    c = (a.broadcast(a_broadcast).template cast<bool>()
         && b.broadcast(b_broadcast).template cast<bool>())
            .template cast<Dst>();
  } else if (binary_op == BinaryOp::kLogicalOr) {
    c = (a.broadcast(a_broadcast).template cast<bool>()
         || b.broadcast(b_broadcast).template cast<bool>())
            .template cast<Dst>();
  } else if (binary_op == BinaryOp::kLogicalXor) {
    c = (a.broadcast(a_broadcast).template cast<bool>()
         ^ b.broadcast(b_broadcast).template cast<bool>())
            .template cast<Dst>();
  } else {
    UNIMPLEMENTED();
  }
  std::vector<int64_t> a_dims = {a.dimension(0), a.dimension(1), a.dimension(2), a.dimension(3)};
  std::vector<int64_t> b_dims = {b.dimension(0), b.dimension(1), b.dimension(2), b.dimension(3)};
  std::vector<int64_t> c_dims = {c.dimension(0), c.dimension(1), c.dimension(2), c.dimension(3)};
  int64_t a_size = a.size() * sizeof(Src);
  int64_t b_size = b.size() * sizeof(Src);
  int64_t c_size = c.size() * sizeof(Dst);

  for (const auto& device_type : device_types) {
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
    if (left_scalar) {
      Src a_value = *reinterpret_cast<Src*>(input_a.ptr());
      binary->Launch(stream.stream(), GetScalar(a_value), num_axes, b_dims.data(), device_b.ptr(),
                     device_c.ptr());
    } else if (right_scalar) {
      Src b_value = *reinterpret_cast<Src*>(input_b.ptr());
      binary->Launch(stream.stream(), num_axes, a_dims.data(), device_a.ptr(), GetScalar(b_value),
                     device_c.ptr());
    } else {
      binary->Launch(stream.stream(), num_axes, a_dims.data(), device_a.ptr(), num_axes,
                     b_dims.data(), device_b.ptr(), device_c.ptr());
    }
    d2h->Launch(stream.stream(), output.ptr(), device_c.ptr(), c_size);
    CHECK_JUST(stream.stream()->Sync());

    Eigen::Map<Eigen::Matrix<Dst, 1, Eigen::Dynamic>, Eigen::Unaligned> eigen_out(c.data(),
                                                                                  c.size());
    Eigen::Map<Eigen::Matrix<Dst, 1, Eigen::Dynamic>, Eigen::Unaligned> of_out(
        reinterpret_cast<Dst*>(output.ptr()), c.size());
    ASSERT_TRUE(eigen_out.template isApprox(of_out));
  }
}

template<BinaryOp binary_op, DataType src_data_type, typename Src, DataType dst_data_type,
         typename Dst>
void TestElementwiseBroadcastBinary(DeviceManagerRegistry* registry,
                                    const std::set<DeviceType>& device_types) {
  TestElementwiseBroadcastBinary<binary_op, src_data_type, Src, dst_data_type, Dst>(
      registry, device_types, 0);
  TestElementwiseBroadcastBinary<binary_op, src_data_type, Src, dst_data_type, Dst>(
      registry, device_types, 1);
  TestElementwiseBroadcastBinary<binary_op, src_data_type, Src, dst_data_type, Dst>(
      registry, device_types, 2);
  TestElementwiseBroadcastBinary<binary_op, src_data_type, Src, dst_data_type, Dst>(
      registry, device_types, 3);
}

template<BinaryOp binary_op>
void TestComputeBinary(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types) {
  TestElementwiseBroadcastBinary<binary_op, DataType::kInt8, int8_t, DataType::kInt8, int8_t>(
      registry, device_types);
  TestElementwiseBroadcastBinary<binary_op, DataType::kUInt8, uint8_t, DataType::kUInt8, uint8_t>(
      registry, device_types);
  TestElementwiseBroadcastBinary<binary_op, DataType::kInt32, int32_t, DataType::kInt32, int32_t>(
      registry, device_types);
  TestElementwiseBroadcastBinary<binary_op, DataType::kInt64, int64_t, DataType::kInt64, int64_t>(
      registry, device_types);
  TestElementwiseBroadcastBinary<binary_op, DataType::kDouble, double, DataType::kDouble, double>(
      registry, device_types);
  TestElementwiseBroadcastBinary<binary_op, DataType::kFloat, float, DataType::kFloat, float>(
      registry, device_types);
  TestElementwiseBroadcastBinary<binary_op, DataType::kFloat16, Eigen::half, DataType::kFloat16,
                                 Eigen::half>(registry, device_types);
}

template<BinaryOp binary_op>
void TestLogicalBinary(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types) {
  TestElementwiseBroadcastBinary<binary_op, DataType::kInt8, int8_t, DataType::kBool, bool>(
      registry, device_types);
  TestElementwiseBroadcastBinary<binary_op, DataType::kUInt8, uint8_t, DataType::kBool, bool>(
      registry, device_types);
  TestElementwiseBroadcastBinary<binary_op, DataType::kInt32, int32_t, DataType::kBool, bool>(
      registry, device_types);
  TestElementwiseBroadcastBinary<binary_op, DataType::kInt64, int64_t, DataType::kBool, bool>(
      registry, device_types);
  TestElementwiseBroadcastBinary<binary_op, DataType::kDouble, double, DataType::kBool, bool>(
      registry, device_types);
  TestElementwiseBroadcastBinary<binary_op, DataType::kFloat, float, DataType::kBool, bool>(
      registry, device_types);
  TestElementwiseBroadcastBinary<binary_op, DataType::kFloat16, Eigen::half, DataType::kBool, bool>(
      registry, device_types);
}

}  // namespace

TEST_F(PrimitiveTest, TestBinary) {
  TestComputeBinary<BinaryOp::kAdd>(&device_manager_registry_, available_device_types_);
  TestComputeBinary<BinaryOp::kSub>(&device_manager_registry_, available_device_types_);
  TestComputeBinary<BinaryOp::kMul>(&device_manager_registry_, available_device_types_);
  TestComputeBinary<BinaryOp::kDiv>(&device_manager_registry_, available_device_types_);
  TestComputeBinary<BinaryOp::kMax>(&device_manager_registry_, available_device_types_);
  TestComputeBinary<BinaryOp::kMin>(&device_manager_registry_, available_device_types_);
  TestLogicalBinary<BinaryOp::kEqual>(&device_manager_registry_, available_device_types_);
  TestLogicalBinary<BinaryOp::kNotEqual>(&device_manager_registry_, available_device_types_);
  TestLogicalBinary<BinaryOp::kLessThan>(&device_manager_registry_, available_device_types_);
  TestLogicalBinary<BinaryOp::kLessEqual>(&device_manager_registry_, available_device_types_);
  TestLogicalBinary<BinaryOp::kGreaterThan>(&device_manager_registry_, available_device_types_);
  TestLogicalBinary<BinaryOp::kGreaterEqual>(&device_manager_registry_, available_device_types_);
  TestLogicalBinary<BinaryOp::kLogicalAnd>(&device_manager_registry_, available_device_types_);
  TestLogicalBinary<BinaryOp::kLogicalOr>(&device_manager_registry_, available_device_types_);
  TestLogicalBinary<BinaryOp::kLogicalXor>(&device_manager_registry_, available_device_types_);
}

}  // namespace test

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow
