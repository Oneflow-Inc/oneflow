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
#include "oneflow/core/ep/include/primitive/constant_pad.h"
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
namespace oneflow {

namespace ep {

namespace primitive {

namespace test {

template<typename T, DataType dtype>
void TestConstantPad2d(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types,
                       const int dims[2], const std::vector<int64_t> padding_before,
                       const std::vector<int64_t> padding_after) {
  using EigenVec = Eigen::Matrix<T, 1, Eigen::Dynamic>;
  int in_elem_cnt = 1;
  int out_elem_cnt = 1;
  for (int i = 0; i < 2; i++) {
    in_elem_cnt *= dims[i];
    out_elem_cnt *= (dims[i] + padding_before[i] + padding_after[i]);
  }
  const int in_matrix_size = in_elem_cnt * sizeof(T);
  const int out_matrix_size = out_elem_cnt * sizeof(T);

  for (const auto& device_type : device_types) {
    Eigen::Tensor<T, 2, Eigen::RowMajor> mat(dims[0], dims[1]);

    mat.setRandom();
    auto device = registry->GetDevice(device_type, 0);

    ep::test::PinnedMemoryGuard host_src(device.get(), in_matrix_size);
    ep::test::PinnedMemoryGuard host_dst(device.get(), out_matrix_size);
    ep::test::DeviceMemoryGuard device_src(device.get(), in_matrix_size);
    ep::test::DeviceMemoryGuard device_dst(device.get(), out_matrix_size);

    ep::test::StreamGuard stream(device.get());
    std::unique_ptr<ConstantPad> constant_pad =
        NewPrimitive<ConstantPadFactory>(device_type, dtype);
    ASSERT_TRUE(constant_pad.operator bool());
    std::unique_ptr<Memcpy> h2d = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kHtoD);
    std::unique_ptr<Memcpy> d2h = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kDtoH);
    ASSERT_TRUE(d2h.operator bool());
    ASSERT_TRUE(h2d.operator bool());
    T* mat_data = mat.data();

    std::memcpy(host_src.ptr(), mat_data, in_matrix_size);
    h2d->Launch(stream.stream(), device_src.ptr<T>(), host_src.ptr<T>(), in_matrix_size);
    const int64_t src_dims[2] = {dims[0], dims[1]};
    constant_pad->Launch(stream.stream(), /*num_dims=*/2, src_dims, device_src.ptr<T>(),
                         padding_before.data(), padding_after.data(), Scalar(0),
                         device_dst.ptr<T>());
    d2h->Launch(stream.stream(), host_dst.ptr<T>(), device_dst.ptr<T>(), out_matrix_size);
    CHECK_JUST(stream.stream()->Sync());

    Eigen::array<std::pair<int, int>, 2> paddings;
    for (int i = 0; i < 2; i++) {
      paddings[i] = std::make_pair(padding_before[i], padding_after[i]);
    }

    Eigen::Tensor<T, 2, Eigen::RowMajor> mat_padded = mat.pad(paddings);
    auto eigen_padded_res = Eigen::Map<EigenVec, Eigen::Unaligned>(
        reinterpret_cast<T*>(mat_padded.data()), out_elem_cnt);
    auto constant_pad_primitive_res =
        Eigen::Map<EigenVec, Eigen::Unaligned>(host_dst.ptr<T>(), out_elem_cnt);
    ASSERT_TRUE(eigen_padded_res.template isApprox(constant_pad_primitive_res));
  }
}

template<typename T, DataType dtype>
void TestConstantPadNegative2d(DeviceManagerRegistry* registry,
                               const std::set<DeviceType>& device_types, const int dims[2],
                               const std::vector<int64_t> padding_before,
                               const std::vector<int64_t> padding_after) {
  using EigenVec = Eigen::Matrix<T, 1, Eigen::Dynamic>;
  int in_elem_cnt = 1;
  int out_elem_cnt = 1;
  int offsets[2];
  int extents[2];

  for (int i = 0; i < 2; i++) {
    in_elem_cnt *= dims[i];
    out_elem_cnt *= (dims[i] + padding_before[i] + padding_after[i]);
    offsets[i] = -padding_before[i];
    extents[i] = dims[i] + padding_before[i] + padding_after[i];
  }
  const int in_matrix_size = in_elem_cnt * sizeof(T);
  const int out_matrix_size = out_elem_cnt * sizeof(T);

  for (const auto& device_type : device_types) {
    Eigen::Tensor<T, 2, Eigen::RowMajor> mat(dims[0], dims[1]);

    mat.setRandom();
    auto device = registry->GetDevice(device_type, 0);

    ep::test::PinnedMemoryGuard host_src(device.get(), in_matrix_size);
    ep::test::PinnedMemoryGuard host_dst(device.get(), out_matrix_size);
    ep::test::DeviceMemoryGuard device_src(device.get(), in_matrix_size);
    ep::test::DeviceMemoryGuard device_dst(device.get(), out_matrix_size);

    ep::test::StreamGuard stream(device.get());
    std::unique_ptr<ConstantPad> constant_pad =
        NewPrimitive<ConstantPadFactory>(device_type, dtype);
    ASSERT_TRUE(constant_pad.operator bool());
    std::unique_ptr<Memcpy> h2d = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kHtoD);
    std::unique_ptr<Memcpy> d2h = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kDtoH);
    ASSERT_TRUE(d2h.operator bool());
    ASSERT_TRUE(h2d.operator bool());
    T* mat_data = mat.data();

    std::memcpy(host_src.ptr(), mat_data, in_matrix_size);
    h2d->Launch(stream.stream(), device_src.ptr<T>(), host_src.ptr<T>(), in_matrix_size);
    const int64_t src_dims[2] = {dims[0], dims[1]};
    constant_pad->Launch(stream.stream(), /*num_dims=*/2, src_dims, device_src.ptr<T>(),
                         padding_before.data(), padding_after.data(), Scalar(0),
                         device_dst.ptr<T>());
    d2h->Launch(stream.stream(), host_dst.ptr<T>(), device_dst.ptr<T>(), out_matrix_size);
    CHECK_JUST(stream.stream()->Sync());

    Eigen::array<Eigen::Index, 2> slice_offsets = {offsets[0], offsets[1]};
    Eigen::array<Eigen::Index, 2> slice_extents = {extents[0], extents[1]};
    Eigen::Tensor<T, 2, Eigen::RowMajor> mat_padded = mat.slice(slice_offsets, slice_extents);
    auto eigen_padded_res = Eigen::Map<EigenVec, Eigen::Unaligned>(
        reinterpret_cast<T*>(mat_padded.data()), out_elem_cnt);
    auto constant_pad_primitive_res =
        Eigen::Map<EigenVec, Eigen::Unaligned>(host_dst.ptr<T>(), out_elem_cnt);
    ASSERT_TRUE(eigen_padded_res.template isApprox(constant_pad_primitive_res));
  }
}

TEST_F(PrimitiveTest, TestConstantPadPrimitive2d) {
  const int32_t dims1[2] = {4, 4};
  const int32_t dims2[2] = {10, 3};
  const int32_t dims3[2] = {31, 4};
  const int32_t dims4[2] = {6, 8};
  const int32_t dims5[2] = {4, 11};

  const std::vector<int64_t> padding_before1 = {1, 1};
  const std::vector<int64_t> padding_after1 = {1, 1};
  const std::vector<int64_t> padding_before2 = {1, 2};
  const std::vector<int64_t> padding_after2 = {2, 1};
  const std::vector<int64_t> padding_before3 = {2, 1};
  const std::vector<int64_t> padding_after3 = {1, 2};
  const std::vector<int64_t> padding_before4 = {3, 1};
  const std::vector<int64_t> padding_after4 = {1, 3};
  const std::vector<int64_t> padding_before5 = {1, 3};
  const std::vector<int64_t> padding_after5 = {3, 1};

  TestConstantPad2d<float, DataType::kFloat>(&device_manager_registry_, available_device_types_,
                                             dims1, padding_before1, padding_after1);
  TestConstantPad2d<double, DataType::kDouble>(&device_manager_registry_, available_device_types_,
                                               dims2, padding_before2, padding_after2);
  TestConstantPad2d<int32_t, DataType::kInt32>(&device_manager_registry_, available_device_types_,
                                               dims3, padding_before3, padding_after3);
  TestConstantPad2d<int64_t, DataType::kInt64>(&device_manager_registry_, available_device_types_,
                                               dims4, padding_before4, padding_after4);
  TestConstantPad2d<Eigen::half, DataType::kFloat16>(
      &device_manager_registry_, available_device_types_, dims5, padding_before5, padding_after5);
}

TEST_F(PrimitiveTest, TestConstantPadPrimitiveNegative2d) {
  // const int32_t dims1[2] = {4, 4};
  const int32_t dims1[2] = {7, 9};

  const int32_t dims2[2] = {10, 7};
  const int32_t dims3[2] = {12, 11};
  const int32_t dims4[2] = {6, 8};
  const int32_t dims5[2] = {4, 11};

  const std::vector<int64_t> padding_before1 = {-1, -1};
  const std::vector<int64_t> padding_after1 = {-1, -1};
  const std::vector<int64_t> padding_before2 = {-2, 0};
  const std::vector<int64_t> padding_after2 = {0, -1};
  const std::vector<int64_t> padding_before3 = {-2, -1};
  const std::vector<int64_t> padding_after3 = {-1, -2};
  const std::vector<int64_t> padding_before4 = {-1, 0};
  const std::vector<int64_t> padding_after4 = {0, -1};
  const std::vector<int64_t> padding_before5 = {-1, -3};
  const std::vector<int64_t> padding_after5 = {0, -1};

  TestConstantPadNegative2d<float, DataType::kFloat>(
      &device_manager_registry_, available_device_types_, dims1, padding_before1, padding_after1);
  TestConstantPadNegative2d<double, DataType::kDouble>(
      &device_manager_registry_, available_device_types_, dims2, padding_before2, padding_after2);
  TestConstantPadNegative2d<int32_t, DataType::kInt32>(
      &device_manager_registry_, available_device_types_, dims3, padding_before3, padding_after3);
  TestConstantPadNegative2d<int64_t, DataType::kInt64>(
      &device_manager_registry_, available_device_types_, dims4, padding_before4, padding_after4);
  TestConstantPadNegative2d<Eigen::half, DataType::kFloat16>(
      &device_manager_registry_, available_device_types_, dims5, padding_before5, padding_after5);
}

}  // namespace test

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow
