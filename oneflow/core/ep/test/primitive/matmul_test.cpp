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
#include "oneflow/core/ep/include/primitive/matmul.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace oneflow {

namespace ep {

namespace primitive {

namespace test {

namespace {

template<DataType data_type, typename T>
void TestMatmul(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types, int m,
                int k, int n, bool transpose_a, bool transpose_b) {
  using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Matrix a = Matrix::Random(m, k);
  Matrix b = Matrix::Random(k, n);
  Matrix c = a * b;
  Matrix a_transpose = a.transpose();
  Matrix b_transpose = b.transpose();

  int64_t a_size = m * k * sizeof(T);
  int64_t b_size = k * n * sizeof(T);
  int64_t c_size = m * n * sizeof(T);

  for (const auto& device_type : device_types) {
    if (device_type == DeviceType::kCPU && data_type == DataType::kFloat16) {
      // CPU matmul not support float16
      continue;
    }
    auto device = registry->GetDevice(device_type, 0);
    ep::test::PinnedMemoryGuard input_a(device.get(), a_size);
    ep::test::PinnedMemoryGuard input_b(device.get(), b_size);
    if (transpose_a) {
      std::memcpy(input_a.ptr(), a_transpose.data(), a_size);
    } else {
      std::memcpy(input_a.ptr(), a.data(), a_size);
    }
    if (transpose_b) {
      std::memcpy(input_b.ptr(), b_transpose.data(), b_size);
    } else {
      std::memcpy(input_b.ptr(), b.data(), b_size);
    }
    ep::test::PinnedMemoryGuard output(device.get(), c_size);
    ep::test::DeviceMemoryGuard device_a(device.get(), a_size);
    ep::test::DeviceMemoryGuard device_b(device.get(), b_size);
    ep::test::DeviceMemoryGuard device_c(device.get(), c_size);
    ep::test::StreamGuard stream(device.get());
    std::unique_ptr<Memcpy> h2d = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kHtoD);
    std::unique_ptr<Memcpy> d2h = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kDtoH);
    const auto trans_a = transpose_a ? BlasTransposeType::T : BlasTransposeType::N;
    const auto trans_b = transpose_b ? BlasTransposeType::T : BlasTransposeType::N;
    std::unique_ptr<Matmul> matmul =
        NewPrimitive<MatmulFactory>(device_type, data_type, trans_a, trans_b);
    ASSERT_TRUE(d2h.operator bool());
    ASSERT_TRUE(h2d.operator bool());
    ASSERT_TRUE(matmul.operator bool());
    h2d->Launch(stream.stream(), device_a.ptr(), input_a.ptr(), a_size);
    h2d->Launch(stream.stream(), device_b.ptr(), input_b.ptr(), b_size);
    matmul->Launch(stream.stream(), m, n, k, 1.0, device_a.ptr(), device_b.ptr(), 0.0,
                   device_c.ptr());
    d2h->Launch(stream.stream(), output.ptr(), device_c.ptr(), c_size);
    CHECK_JUST(stream.stream()->Sync());
    auto res = Eigen::Map<Matrix, Eigen::Unaligned>(reinterpret_cast<T*>(output.ptr()), m, n);
    ASSERT_TRUE(c.template isApprox(res, static_cast<T>(0.001)));
  }
}

template<DataType data_type, typename T>
void TestMatmul(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types, int m,
                int k, int n) {
  TestMatmul<data_type, T>(registry, device_types, m, k, n, false, false);
  TestMatmul<data_type, T>(registry, device_types, m, k, n, true, false);
  TestMatmul<data_type, T>(registry, device_types, m, k, n, false, true);
  TestMatmul<data_type, T>(registry, device_types, m, k, n, true, true);
}

template<DataType data_type, typename T>
void TestMatmul(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types) {
  TestMatmul<data_type, T>(registry, device_types, 64, 16, 8);
  TestMatmul<data_type, T>(registry, device_types, 16, 7, 12);
}

}  // namespace

TEST_F(PrimitiveTest, TestMatmul) {
  TestMatmul<DataType::kDouble, double>(&device_manager_registry_, available_device_types_);
  TestMatmul<DataType::kFloat, float>(&device_manager_registry_, available_device_types_);
  TestMatmul<DataType::kFloat16, Eigen::half>(&device_manager_registry_, available_device_types_);
}

}  // namespace test

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow
