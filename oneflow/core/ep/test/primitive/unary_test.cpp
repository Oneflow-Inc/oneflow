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
#include "oneflow/core/ep/include/primitive/elementwise_unary.h"
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
                                    const std::set<DeviceType>& device_types) {
  const int num_axes = 4;
  const int broadcast_dim0 = 2;
  const int broadcast_dim1 = 3;
  const int broadcast_dim2 = 2;
  const int broadcast_dim3 = 4;

  const int a_dim0 = 1;
  const int a_dim1 = broadcast_dim1;
  const int a_dim2 = broadcast_dim2;
  const int a_dim3 = broadcast_dim3;
  const int a_broadcast0 = broadcast_dim0;
  const int a_broadcast1 = 1;
  const int a_broadcast2 = 1;
  const int a_broadcast3 = 1;
  const Eigen::array<int, 4> a_broadcast = {a_broadcast0, a_broadcast1, a_broadcast2, a_broadcast3};
  Eigen::Tensor<Src, 4, Eigen::RowMajor> a(a_dim0, a_dim1, a_dim2, a_dim3);

  const std::vector<int64_t> a_strides = {a_dim1*a_dim2*a_dim3, a_dim2*a_dim3, a_dim3, 1};
  const std::vector<int64_t> c_strides = {broadcast_dim2*broadcast_dim3, broadcast_dim0*broadcast_dim2*broadcast_dim3, 1, broadcast_dim2};

  a.setRandom();
  std::vector<int64_t> a_dims = {a.dimension(0), a.dimension(1), a.dimension(2), a.dimension(3)};
  std::vector<int64_t> c_dims = {broadcast_dim0, broadcast_dim1, broadcast_dim2, broadcast_dim3};

  Eigen::Tensor<Dst, 4, Eigen::RowMajor> broadcast_a = a.broadcast(a_broadcast).template cast<Dst>();
  
  const int64_t a_size           = a.size() * sizeof(Src);
  const int64_t c_count          = std::accumulate(c_dims.begin(), c_dims.end(), 1, std::multiplies<int64_t>());
  const int64_t c_size           = c_count * sizeof(Dst);
  const int64_t broadcast_a_size = broadcast_a.size() * sizeof(Dst);

  ASSERT_TRUE(c_size == broadcast_a_size);

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

    unary->Launch(stream.stream(), device_broadcast_a.ptr(), device_broadcast_c.ptr(), c_count); // c.size() is for count

    d2h->Launch(stream.stream(), broadcast_output.ptr(), device_broadcast_c.ptr(), c_size); // c_size is in bytes
    CHECK_JUST(stream.stream()->Sync());

    ep::test::PinnedMemoryGuard input_a(device.get(), a_size);
    std::memcpy(input_a.ptr(), a.data(), a_size);

    ep::test::PinnedMemoryGuard output(device.get(), c_size);
    ep::test::DeviceMemoryGuard device_a(device.get(), a_size);
    ep::test::DeviceMemoryGuard device_c(device.get(), c_size);
    std::unique_ptr<BroadcastElementwiseUnary> broadcast_unary =
        NewPrimitive<BroadcastElementwiseUnaryFactory>(device_type, unary_op, src_data_type,
                                                        dst_data_type, num_axes);
    ASSERT_TRUE(broadcast_unary.operator bool());
    h2d->Launch(stream.stream(), device_a.ptr(), input_a.ptr(), a_size);

    broadcast_unary->Launch(stream.stream(), num_axes, a_dims.data(), a_strides.data(), device_a.ptr(),
      num_axes, c_dims.data(), c_strides.data(), device_c.ptr());

    d2h->Launch(stream.stream(), output.ptr(), device_c.ptr(), c_size);
    CHECK_JUST(stream.stream()->Sync());

    Dst thresh = 1e-4;
    bool res = true;

    for (int i0 = 0; i0 < broadcast_dim0; i0++) {
        for (int i1 = 0; i1 < broadcast_dim1; i1++) {
            for (int i2 = 0; i2 < broadcast_dim2; i2++) {
                for (int i3 = 0; i3 < broadcast_dim3; i3++) {
                    #define ABS(x) ((x > 0) ? (x) : (-x))
                    const size_t src_index = broadcast_dim1*broadcast_dim2*broadcast_dim3*i0 + broadcast_dim2*broadcast_dim3*i1
                        + broadcast_dim3*i2 + i3;
                    const size_t dst_index = broadcast_dim2*broadcast_dim3*i0 + broadcast_dim0*broadcast_dim2*broadcast_dim3*i1 + i2 + broadcast_dim2*i3;
                    if (ABS(reinterpret_cast<Dst*>(broadcast_output.ptr())[src_index] - reinterpret_cast<Dst*>(output.ptr())[dst_index]) > thresh) {
                        res = false;
                    }
                }
            }
        }
    }
    ASSERT_TRUE(res);
  }
}

template<UnaryOp unary_op>
void TestComputeUnary(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types) {
  /*
  TestElementwiseBroadcastUnary<unary_op, DataType::kInt8, int8_t, DataType::kInt8, int8_t>(
      registry, device_types);
  TestElementwiseBroadcastUnary<unary_op, DataType::kUInt8, uint8_t, DataType::kUInt8, uint8_t>(
      registry, device_types);
  TestElementwiseBroadcastUnary<unary_op, DataType::kInt32, int32_t, DataType::kInt32, int32_t>(
      registry, device_types);
  TestElementwiseBroadcastUnary<unary_op, DataType::kInt64, int64_t, DataType::kInt64, int64_t>(
      registry, device_types);
  TestElementwiseBroadcastUnary<unary_op, DataType::kDouble, double, DataType::kDouble, double>(
      registry, device_types);*/
  TestElementwiseBroadcastUnary<unary_op, DataType::kFloat, float, DataType::kFloat, float>(
      registry, device_types);
  /*
  TestElementwiseBroadcastUnary<unary_op, DataType::kFloat16, Eigen::half, DataType::kFloat16,
                                 Eigen::half>(registry, device_types);*/
}

template<UnaryOp unary_op>
void TestLogicalUnary(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types) {
  /*
  TestElementwiseBroadcastUnary<unary_op, DataType::kInt8, int8_t, DataType::kBool, bool>(
      registry, device_types);
  TestElementwiseBroadcastUnary<unary_op, DataType::kUInt8, uint8_t, DataType::kBool, bool>(
      registry, device_types);
  TestElementwiseBroadcastUnary<unary_op, DataType::kInt32, int32_t, DataType::kBool, bool>(
      registry, device_types);
  TestElementwiseBroadcastUnary<unary_op, DataType::kInt64, int64_t, DataType::kBool, bool>(
      registry, device_types);
  TestElementwiseBroadcastUnary<unary_op, DataType::kDouble, double, DataType::kBool, bool>(
      registry, device_types);*/
  TestElementwiseBroadcastUnary<unary_op, DataType::kFloat, float, DataType::kBool, bool>(
      registry, device_types);
  /*
  TestElementwiseBroadcastUnary<unary_op, DataType::kFloat16, Eigen::half, DataType::kBool, bool>(
      registry, device_types);*/
}

}  // namespace

TEST_F(PrimitiveTest, TestUnary) {
  TestComputeUnary<UnaryOp::kAbs>(&device_manager_registry_, available_device_types_);
}

}  // namespace test

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow
