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
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_unary.h"
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

template<UnaryOp unary_op, DataType src_data_type, typename Src, DataType dst_data_type,
         typename Dst>
void TestElementwiseBroadcastUnary(DeviceManagerRegistry* registry,
                                    const std::set<DeviceType>& device_types, int test_type) {
  const int num_axes = 4;
  const int broadcast_dim0 = 16;
  const int broadcast_dim1 = 3;
  const int broadcast_dim2 = 4;
  const int broadcast_dim3 = 8;
  bool is_broadcast = true;

  /*
  if (test_type == 0) {
    // do nothing
  } else if (test_type == 1) {
    is_broadcast = true;
  } else {
    UNIMPLEMENTED();
  }*/
  //TODO: input & output could have different num_dims, now they are the same
  const int a_dim0 = broadcast_dim0;
  const int a_dim1 = broadcast_dim1;
  const int a_dim2 = broadcast_dim2;
  const int a_dim3 = is_broadcast ? 1 : broadcast_dim3;
  const int a_broadcast0 = 1;
  const int a_broadcast1 = 1;
  const int a_broadcast2 = 1;
  const int a_broadcast3 = is_broadcast ? broadcast_dim3 : 1;
  const Eigen::array<int, 4> a_broadcast = {a_broadcast0, a_broadcast1, a_broadcast2, a_broadcast3};
  Eigen::Tensor<Src, 4, Eigen::RowMajor> a(a_dim0, a_dim1, a_dim2, a_dim3);
  Eigen::Tensor<Dst, 4, Eigen::RowMajor> c(broadcast_dim0, broadcast_dim1, broadcast_dim2,
                                           broadcast_dim3);
  auto broadcast_a = a.broadcast(a_broadcast);

  a.setRandom();
  std::vector<int64_t> a_dims = {a.dimension(0), a.dimension(1), a.dimension(2), a.dimension(3)};
  std::vector<int64_t> c_dims = {c.dimension(0), c.dimension(1), c.dimension(2), c.dimension(3)};
  
  const int64_t a_size           = a.size() * sizeof(Src);
  const int64_t c_size           = c.size() * sizeof(Dst);
  const int64_t broadcast_a_size = broadcast_a.size() * sizeof(Src);

  for (const auto& device_type : device_types) {
    // broadcast a with non-broadcast elementwise unary primitive
    auto device = registry->GetDevice(device_type, 0);
    ep::test::PinnedMemoryGuard input_broadcast_a(device.get(), broadcast_a_size);
    std::memcpy(input_broadcast_a.ptr(), broadcast_a.data(), broadcast_a_size);

    ep::test::PinnedMemoryGuard broadcast_output(device.get(), c_size);
    ep::test::DeviceMemoryGuard device_broadcast_a(device.get(), broadcast_a_size);
    ep::test::DeviceMemoryGuard device_broadcast_c(device.get(), c_size);
    ep::test::StreamGuard stream(device.get());
    std::unique_ptr<Memcpy> h2d = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kHtoD);
    std::unique_ptr<Memcpy> d2h = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kDtoH);
    std::unique_ptr<ElementwiseUnary> unary =
        NewPrimitive<ElementwiseUnaryFactory>(device_type, unary_op, src_data_type,
                                                        dst_data_type);
    ASSERT_TRUE(d2h.operator bool());
    ASSERT_TRUE(h2d.operator bool());
    ASSERT_TRUE(unary.operator bool());
    h2d->Launch(stream.stream(), device_broadcast_a.ptr(), input_broadcast_a.ptr(), broadcast_a_size);

    unary->Launch(stream.stream(), device_broadcast_a.ptr(), device_c.data(), c_size);

    d2h->Launch(stream.stream(), broadcast_output.ptr(), device_c.ptr(), c_size);
    CHECK_JUST(stream.stream()->Sync());


    ep::test::PinnedMemoryGuard input_a(device.get(), a_size);
    std::memcpy(input_a.ptr(), a.data(), a_size);

    ep::test::PinnedMemoryGuard output(device.get(), c_size);
    ep::test::DeviceMemoryGuard device_a(device.get(), a_size);
    ep::test::DeviceMemoryGuard device_c(device.get(), c_size);
    ep::test::StreamGuard stream(device.get());
    std::unique_ptr<BroadcastElementwiseUnary> broadcast_unary =
        NewPrimitive<BroadcastElementwiseUnaryFactory>(device_type, unary_op, src_data_type,
                                                        dst_data_type, num_axes);
    ASSERT_TRUE(broadcast_unary.operator bool());
    h2d->Launch(stream.stream(), device_a.ptr(), input_a.ptr(), a_size);

    broadcast_unary->Launch(stream.stream(), num_axes, a_dims.data(), device_a.ptr(),
      num_axes, c_dims.data(), device_c.data());

    d2h->Launch(stream.stream(), output.ptr(), device_c.ptr(), c_size);
    CHECK_JUST(stream.stream()->Sync());

    Eigen::Map<Eigen::Matrix<Dst, 1, Eigen::Dynamic>, Eigen::Unaligned> eigen_out
      (reinterpret_cast<Dst*>(broadcast_output.ptr()), c.size());
    Eigen::Map<Eigen::Matrix<Dst, 1, Eigen::Dynamic>, Eigen::Unaligned> of_out(
        reinterpret_cast<Dst*>(output.ptr()), c.size());
    ASSERT_TRUE(eigen_out.template isApprox(of_out));
  }
}

template<UnaryOp unary_op, DataType src_data_type, typename Src, DataType dst_data_type,
         typename Dst>
void TestElementwiseBroadcastUnary(DeviceManagerRegistry* registry,
                                    const std::set<DeviceType>& device_types) {
  TestElementwiseBroadcastUnary<unary_op, src_data_type, Src, dst_data_type, Dst>(
      registry, device_types, 0);
  TestElementwiseBroadcastUnary<unary_op, src_data_type, Src, dst_data_type, Dst>(
      registry, device_types, 1);
  TestElementwiseBroadcastUnary<unary_op, src_data_type, Src, dst_data_type, Dst>(
      registry, device_types, 2);
  TestElementwiseBroadcastUnary<unary_op, src_data_type, Src, dst_data_type, Dst>(
      registry, device_types, 3);
}

template<UnaryOp unary_op>
void TestComputeUnary(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types) {
  TestElementwiseBroadcastUnary<unary_op, DataType::kInt8, int8_t, DataType::kInt8, int8_t>(
      registry, device_types);
  TestElementwiseBroadcastUnary<unary_op, DataType::kUInt8, uint8_t, DataType::kUInt8, uint8_t>(
      registry, device_types);
  TestElementwiseBroadcastUnary<unary_op, DataType::kInt32, int32_t, DataType::kInt32, int32_t>(
      registry, device_types);
  TestElementwiseBroadcastUnary<unary_op, DataType::kInt64, int64_t, DataType::kInt64, int64_t>(
      registry, device_types);
  TestElementwiseBroadcastUnary<unary_op, DataType::kDouble, double, DataType::kDouble, double>(
      registry, device_types);
  TestElementwiseBroadcastUnary<unary_op, DataType::kFloat, float, DataType::kFloat, float>(
      registry, device_types);
  TestElementwiseBroadcastUnary<unary_op, DataType::kFloat16, Eigen::half, DataType::kFloat16,
                                 Eigen::half>(registry, device_types);
}

template<UnaryOp unary_op>
void TestLogicalUnary(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types) {
  TestElementwiseBroadcastUnary<unary_op, DataType::kInt8, int8_t, DataType::kBool, bool>(
      registry, device_types);
  TestElementwiseBroadcastUnary<unary_op, DataType::kUInt8, uint8_t, DataType::kBool, bool>(
      registry, device_types);
  TestElementwiseBroadcastUnary<unary_op, DataType::kInt32, int32_t, DataType::kBool, bool>(
      registry, device_types);
  TestElementwiseBroadcastUnary<unary_op, DataType::kInt64, int64_t, DataType::kBool, bool>(
      registry, device_types);
  TestElementwiseBroadcastUnary<unary_op, DataType::kDouble, double, DataType::kBool, bool>(
      registry, device_types);
  TestElementwiseBroadcastUnary<unary_op, DataType::kFloat, float, DataType::kBool, bool>(
      registry, device_types);
  TestElementwiseBroadcastUnary<unary_op, DataType::kFloat16, Eigen::half, DataType::kBool, bool>(
      registry, device_types);
}

}  // namespace

TEST_F(PrimitiveTest, TestUnary) {
  TestComputeUnary<BinaryOp::kAdd>(&device_manager_registry_, available_device_types_);
  TestComputeUnary<BinaryOp::kSub>(&device_manager_registry_, available_device_types_);
  TestComputeUnary<BinaryOp::kMul>(&device_manager_registry_, available_device_types_);
  TestComputeUnary<BinaryOp::kDiv>(&device_manager_registry_, available_device_types_);
  TestComputeUnary<BinaryOp::kMax>(&device_manager_registry_, available_device_types_);
  TestComputeUnary<BinaryOp::kMin>(&device_manager_registry_, available_device_types_);
  TestLogicalUnary<BinaryOp::kEqual>(&device_manager_registry_, available_device_types_);
  TestLogicalUnary<BinaryOp::kNotEqual>(&device_manager_registry_, available_device_types_);
  TestLogicalUnary<BinaryOp::kLessThan>(&device_manager_registry_, available_device_types_);
  TestLogicalUnary<BinaryOp::kLessEqual>(&device_manager_registry_, available_device_types_);
  TestLogicalUnary<BinaryOp::kGreaterThan>(&device_manager_registry_, available_device_types_);
  TestLogicalUnary<BinaryOp::kGreaterEqual>(&device_manager_registry_, available_device_types_);
  TestLogicalUnary<BinaryOp::kLogicalAnd>(&device_manager_registry_, available_device_types_);
  TestLogicalUnary<BinaryOp::kLogicalOr>(&device_manager_registry_, available_device_types_);
  TestLogicalUnary<BinaryOp::kLogicalXor>(&device_manager_registry_, available_device_types_);
}

}  // namespace test

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow
