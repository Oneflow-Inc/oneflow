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

template<UnaryOp unary_op, DataType src_data_type, typename Src, DataType dst_data_type,
         typename Dst>
void TestElementwiseBroadcastUnary(DeviceManagerRegistry* registry,
                                   const std::set<DeviceType>& device_types) {
  const std::vector<int> num_src_axes = {1, 4, 1, 4, 4};
  const std::vector<int> num_dst_axes = {4, 4, 1, 4, 4};

  const std::vector<std::vector<int64_t>> a_dims_vec = {
      {1, 1, 1, 1}, {1, 3, 2, 4}, {1, 1, 1, 1}, {1, 2, 3, 4}, {1, 2, 3, 4}};
  const std::vector<std::vector<int64_t>> broadcast_dims_vec = {
      {2, 3, 2, 4}, {2, 3, 2, 4}, {1, 1, 1, 1}, {1, 2, 3, 4}, {1, 2, 3, 4}};
  const std::vector<std::vector<int64_t>> a_broadcasts_vec = {
      {2, 3, 2, 4}, {2, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}};

  const std::vector<std::vector<int64_t>> a_strides_vec = {
      {0, 0, 0, 0},
      {a_dims_vec[1][1] * a_dims_vec[1][2] * a_dims_vec[1][3], a_dims_vec[1][2] * a_dims_vec[1][3],
       a_dims_vec[1][3], 1},
      {0, 0, 0, 0},
      {a_dims_vec[3][1] * a_dims_vec[3][2] * a_dims_vec[3][3], a_dims_vec[3][2] * a_dims_vec[3][3],
       a_dims_vec[3][3], 1},
      {a_dims_vec[4][1] * a_dims_vec[4][2] * a_dims_vec[4][3], a_dims_vec[4][2] * a_dims_vec[4][3],
       a_dims_vec[4][3], 1}};
  const std::vector<std::vector<int64_t>> c_strides_vec = {
      {broadcast_dims_vec[0][1] * broadcast_dims_vec[0][2] * broadcast_dims_vec[0][3],
       broadcast_dims_vec[0][2] * broadcast_dims_vec[0][3], broadcast_dims_vec[0][3], 1},
      {broadcast_dims_vec[1][2] * broadcast_dims_vec[1][3],
       broadcast_dims_vec[1][0] * broadcast_dims_vec[1][2] * broadcast_dims_vec[1][3], 1,
       broadcast_dims_vec[1][2]},
      {0, 0, 0, 0},
      {broadcast_dims_vec[3][1] * broadcast_dims_vec[3][2] * broadcast_dims_vec[3][3],
       broadcast_dims_vec[3][2], 1, broadcast_dims_vec[3][1] * broadcast_dims_vec[3][2]},
      {1, broadcast_dims_vec[4][0], broadcast_dims_vec[4][0] * broadcast_dims_vec[4][1],
       broadcast_dims_vec[4][0] * broadcast_dims_vec[4][1] * broadcast_dims_vec[4][2]}};

  for (int i = 0; i < 5; i++) {
    const std::vector<int64_t>& a_dims = a_dims_vec[i];
    const std::vector<int64_t>& c_dims = broadcast_dims_vec[i];
    const Eigen::array<int64_t, 4> a_broadcast = {a_broadcasts_vec[i][0], a_broadcasts_vec[i][1],
                                                  a_broadcasts_vec[i][2], a_broadcasts_vec[i][3]};
    Eigen::Tensor<Src, 4, Eigen::RowMajor> a(a_dims[0], a_dims[1], a_dims[2], a_dims[3]);

    const std::vector<int64_t>& a_strides = a_strides_vec[i];
    const std::vector<int64_t>& c_strides = c_strides_vec[i];

    a.setRandom();

    Eigen::Tensor<Src, 4, Eigen::RowMajor> t = a.broadcast(a_broadcast);
    Eigen::Tensor<Dst, 4, Eigen::RowMajor> broadcast_a = t.template cast<Dst>();

    const int64_t a_size = a.size() * sizeof(Src);
    const int64_t c_count =
        std::accumulate(c_dims.begin(), c_dims.end(), 1, std::multiplies<int64_t>());
    const int64_t c_size = c_count * sizeof(Dst);
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
      std::unique_ptr<ElementwiseUnary> unary = NewPrimitive<ElementwiseUnaryFactory>(
          device_type, unary_op, src_data_type, dst_data_type);
      ASSERT_TRUE(d2h.operator bool());
      ASSERT_TRUE(h2d.operator bool());
      ASSERT_TRUE(unary.operator bool());
      h2d->Launch(stream.stream(), device_broadcast_a.ptr(), input_broadcast_a.ptr(),
                  broadcast_a_size);
      unary->Launch(stream.stream(), device_broadcast_a.ptr(), device_broadcast_c.ptr(),
                    c_count);  // c.size() is for count
      d2h->Launch(stream.stream(), broadcast_output.ptr(), device_broadcast_c.ptr(),
                  c_size);  // c_size is in bytes
      CHECK_JUST(stream.stream()->Sync());

      ep::test::PinnedMemoryGuard input_a(device.get(), a_size);
      std::memcpy(input_a.ptr(), a.data(), a_size);

      ep::test::PinnedMemoryGuard output(device.get(), c_size);
      ep::test::DeviceMemoryGuard device_a(device.get(), a_size);
      ep::test::DeviceMemoryGuard device_c(device.get(), c_size);
      std::unique_ptr<BroadcastElementwiseUnary> broadcast_unary =
          NewPrimitive<BroadcastElementwiseUnaryFactory>(device_type, unary_op, src_data_type,
                                                         dst_data_type,
                                                         MAX(num_src_axes[i], num_dst_axes[i]));
      ASSERT_TRUE(broadcast_unary.operator bool());
      h2d->Launch(stream.stream(), device_a.ptr(), input_a.ptr(), a_size);

      broadcast_unary->Launch(stream.stream(), num_src_axes[i], a_dims.data(), a_strides.data(),
                              device_a.ptr(), num_dst_axes[i], c_dims.data(), c_strides.data(),
                              device_c.ptr());
      d2h->Launch(stream.stream(), output.ptr(), device_c.ptr(), c_size);
      CHECK_JUST(stream.stream()->Sync());

      Dst thresh = 1e-4;
      bool res = true;

      std::vector<int64_t> a_broadcast_strides;
      for (int j = num_dst_axes[i] - 1; j >= 0; j--) {
        if (j == num_dst_axes[i] - 1) {
          a_broadcast_strides.push_back(1);
        } else {
          a_broadcast_strides.insert(a_broadcast_strides.begin(),
                                     a_broadcast_strides[0] * a_dims[j + 1] * a_broadcast[j + 1]);
        }
      }

      for (int i0 = 0; i0 < c_dims[0]; i0++) {
        for (int i1 = 0; i1 < c_dims[1]; i1++) {
          for (int i2 = 0; i2 < c_dims[2]; i2++) {
            for (int i3 = 0; i3 < c_dims[3]; i3++) {
#define ABS(x) ((x > 0) ? (x) : (-x))
              const size_t src_index = a_broadcast_strides[0] * i0 + a_broadcast_strides[1] * i1
                                       + a_broadcast_strides[2] * i2 + a_broadcast_strides[3] * i3;
              const size_t dst_index =
                  c_strides[0] * i0 + c_strides[1] * i1 + c_strides[2] * i2 + c_strides[3] * i3;
              if (ABS(reinterpret_cast<Dst*>(broadcast_output.ptr())[src_index]
                      - reinterpret_cast<Dst*>(output.ptr())[dst_index])
                  > thresh) {
                res = false;
              }
#undef ABS
            }
          }
        }
      }
      ASSERT_TRUE(res);
    }
  }
}

template<DataType src_data_type, typename Src, DataType dst_data_type, typename Dst>
void TestElementwiseBroadcastUnaryBatchPermute(DeviceManagerRegistry* registry,
                                               const std::set<DeviceType>& device_types) {
  const std::vector<int64_t>& a_dims = {5, 2};
  const std::vector<int64_t>& c_dims = {5, 2};
  Eigen::Tensor<Src, 2, Eigen::RowMajor> a(5, 4);

  const std::vector<std::vector<int64_t>>& a_strides = {{4, 1}, {2, 1}};
  const std::vector<std::vector<int64_t>>& c_strides = {{1, 5}, {1, 10}};

  a.setRandom();

  const int64_t a_size = a.size() * sizeof(Src);
  const int64_t c_count =
      std::accumulate(c_dims.begin(), c_dims.end(), 1, std::multiplies<int64_t>());
  const int64_t c_size = MAX(c_count, a.size()) * sizeof(Dst);

  for (int i = 0; i < a_strides.size(); i++) {
    auto& a_stride = a_strides[i];
    auto& c_stride = c_strides[i];
    for (const auto& device_type : device_types) {
      // broadcast a with non-broadcast elementwise unary primitive
      auto device = registry->GetDevice(device_type, 0);
      ep::test::StreamGuard stream(device.get());

      ep::test::PinnedMemoryGuard input_a(device.get(), a_size);
      std::memcpy(input_a.ptr(), a.data(), a_size);

      ep::test::PinnedMemoryGuard output(device.get(), c_size);
      ep::test::DeviceMemoryGuard device_a(device.get(), a_size);
      ep::test::DeviceMemoryGuard device_c(device.get(), c_size);
      std::unique_ptr<Memcpy> h2d = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kHtoD);
      std::unique_ptr<Memcpy> d2h = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kDtoH);
      std::unique_ptr<BroadcastElementwiseUnary> broadcast_unary =
          NewPrimitive<BroadcastElementwiseUnaryFactory>(device_type, UnaryOp::kIdentity,
                                                         src_data_type, dst_data_type, 2);
      ASSERT_TRUE(broadcast_unary.operator bool());
      ASSERT_TRUE(d2h.operator bool());
      ASSERT_TRUE(h2d.operator bool());
      h2d->Launch(stream.stream(), device_a.ptr(), input_a.ptr(), a_size);

      broadcast_unary->Launch(stream.stream(), 2, a_dims.data(), a_stride.data(), device_a.ptr(), 2,
                              c_dims.data(), c_stride.data(), device_c.ptr());

      d2h->Launch(stream.stream(), output.ptr(), device_c.ptr(), c_size);
      CHECK_JUST(stream.stream()->Sync());

      Dst thresh = 1e-4;
      bool res = true;

      for (int i0 = 0; i0 < c_dims[0]; i0++) {
        for (int i1 = 0; i1 < c_dims[1]; i1++) {
#define ABS(x) ((x > 0) ? (x) : (-x))
          const size_t src_index = a_stride[0] * i0 + a_stride[1] * i1;
          const size_t dst_index = c_stride[0] * i0 + c_stride[1] * i1;
          if (ABS(reinterpret_cast<Dst*>(input_a.ptr())[src_index]
                  - reinterpret_cast<Dst*>(output.ptr())[dst_index])
              > thresh) {
            res = false;
          }
#undef ABS
        }
      }
      ASSERT_TRUE(res);
    }
  }
}

}  // namespace

TEST_F(PrimitiveTest, TestUnary) {
  TestElementwiseBroadcastUnary<UnaryOp::kIdentity, DataType::kFloat, float, DataType::kFloat,
                                float>(&device_manager_registry_, available_device_types_);
  TestElementwiseBroadcastUnaryBatchPermute<DataType::kFloat, float, DataType::kFloat, float>(
      &device_manager_registry_, available_device_types_);
}

}  // namespace test

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow
