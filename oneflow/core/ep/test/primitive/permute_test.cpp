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
#include "oneflow/core/ep/include/primitive/permute.h"
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
namespace oneflow {

namespace ep {

namespace primitive {

namespace test {

template<typename T, DataType dtype, int NumDims>
void TestPermute2D(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types,
                   const int dims[NumDims], const int permutation_list[NumDims]) {
  using EigenVec = Eigen::Matrix<T, 1, Eigen::Dynamic>;
  const int elem_cnt = dims[0] * dims[1];
  const int matrix_size = elem_cnt * sizeof(T);

  for (const auto& device_type : device_types) {
    Eigen::Tensor<T, NumDims, Eigen::RowMajor> mat(dims[0], dims[1]);
    mat.setRandom();
    auto device = registry->GetDevice(device_type, 0);

    ep::test::PinnedMemoryGuard host_src(device.get(), matrix_size);
    ep::test::PinnedMemoryGuard host_dst(device.get(), matrix_size);
    ep::test::DeviceMemoryGuard device_src(device.get(), matrix_size);
    ep::test::DeviceMemoryGuard device_dst(device.get(), matrix_size);

    ep::test::StreamGuard stream(device.get());
    std::unique_ptr<Permute> permute =
        NewPrimitive<PermuteFactory>(device_type, /*max_num_dims=*/NumDims);
    ASSERT_TRUE(permute.operator bool());
    std::unique_ptr<Memcpy> h2d = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kHtoD);
    std::unique_ptr<Memcpy> d2h = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kDtoH);
    ASSERT_TRUE(d2h.operator bool());
    ASSERT_TRUE(h2d.operator bool());
    T* mat_data = mat.data();
    std::memcpy(host_src.ptr(), mat_data, matrix_size);
    h2d->Launch(stream.stream(), device_src.ptr<T>(), host_src.ptr<T>(), matrix_size);
    const int64_t src_dims[NumDims] = {dims[0], dims[1]};
    permute->Launch(stream.stream(), dtype, /*num_dims=*/NumDims, src_dims, device_src.ptr<T>(),
                    permutation_list, device_dst.ptr<T>());
    d2h->Launch(stream.stream(), host_dst.ptr<T>(), device_dst.ptr<T>(), matrix_size);
    CHECK_JUST(stream.stream()->Sync());

    Eigen::array<int, NumDims> shuffle_index({permutation_list[0], permutation_list[1]});
    Eigen::Tensor<T, NumDims, Eigen::RowMajor> mat_transposed = mat.shuffle(shuffle_index);

    auto eigen_transposed_res = Eigen::Map<EigenVec, Eigen::Unaligned>(
        reinterpret_cast<T*>(mat_transposed.data()), elem_cnt);
    auto permute_primitive_res =
        Eigen::Map<EigenVec, Eigen::Unaligned>(host_dst.ptr<T>(), elem_cnt);
    ASSERT_TRUE(eigen_transposed_res.template isApprox(permute_primitive_res));
  }
}

template<typename T, DataType dtype, int NumDims>
void TestPermute3D(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types,
                   const int dims[NumDims], const int permutation_list[NumDims]) {
  using EigenVec = Eigen::Matrix<T, 1, Eigen::Dynamic>;
  const int elem_cnt = dims[0] * dims[1] * dims[2];
  const int matrix_size = elem_cnt * sizeof(T);

  for (const auto& device_type : device_types) {
    Eigen::Tensor<T, NumDims, Eigen::RowMajor> mat(dims[0], dims[1], dims[2]);
    mat.setRandom();
    auto device = registry->GetDevice(device_type, 0);

    ep::test::PinnedMemoryGuard host_src(device.get(), matrix_size);
    ep::test::PinnedMemoryGuard host_dst(device.get(), matrix_size);
    ep::test::DeviceMemoryGuard device_src(device.get(), matrix_size);
    ep::test::DeviceMemoryGuard device_dst(device.get(), matrix_size);

    ep::test::StreamGuard stream(device.get());
    std::unique_ptr<Permute> permute =
        NewPrimitive<PermuteFactory>(device_type, /*max_num_dims=*/NumDims);
    ASSERT_TRUE(permute.operator bool());
    std::unique_ptr<Memcpy> h2d = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kHtoD);
    std::unique_ptr<Memcpy> d2h = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kDtoH);
    ASSERT_TRUE(d2h.operator bool());
    ASSERT_TRUE(h2d.operator bool());
    T* mat_data = mat.data();
    std::memcpy(host_src.ptr(), mat_data, matrix_size);
    h2d->Launch(stream.stream(), device_src.ptr<T>(), host_src.ptr<T>(), matrix_size);
    const int64_t src_dims[NumDims] = {dims[0], dims[1], dims[2]};
    permute->Launch(stream.stream(), dtype, /*num_dims=*/NumDims, src_dims, device_src.ptr<T>(),
                    permutation_list, device_dst.ptr<T>());
    d2h->Launch(stream.stream(), host_dst.ptr<T>(), device_dst.ptr<T>(), matrix_size);
    CHECK_JUST(stream.stream()->Sync());

    Eigen::array<int, NumDims> shuffle_index(
        {permutation_list[0], permutation_list[1], permutation_list[2]});
    Eigen::Tensor<T, NumDims, Eigen::RowMajor> mat_transposed = mat.shuffle(shuffle_index);

    auto eigen_transposed_res = Eigen::Map<EigenVec, Eigen::Unaligned>(
        reinterpret_cast<T*>(mat_transposed.data()), elem_cnt);
    auto permute_primitive_res =
        Eigen::Map<EigenVec, Eigen::Unaligned>(host_dst.ptr<T>(), elem_cnt);
    ASSERT_TRUE(eigen_transposed_res.template isApprox(permute_primitive_res));
  }
}

TEST_F(PrimitiveTest, TestBatchPermute) {
  const int permutation_list[2] = {1, 0};
  const int32_t dims0[2] = {1024*256, 256};

  TestPermute2D<float, DataType::kFloat, 2>(&device_manager_registry_, available_device_types_,
                                            dims0, permutation_list);
}

}  // namespace test

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow
