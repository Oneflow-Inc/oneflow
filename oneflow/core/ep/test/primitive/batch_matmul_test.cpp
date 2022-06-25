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
#include "oneflow/core/ep/include/primitive/batch_matmul.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace oneflow {

namespace ep {

namespace primitive {

namespace test {

namespace {

template<DataType data_type, typename T>
void TestBatchMatmul(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types,
                     int batch_size, int m, int k, int n, bool transpose_a, bool transpose_b) {
  using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Tensor<T, 3, Eigen::RowMajor> in_a_buffer(batch_size, m, k);
  Eigen::Tensor<T, 3, Eigen::RowMajor> in_b_buffer(batch_size, k, n);
  Eigen::Tensor<T, 3, Eigen::RowMajor> out_c_buffer(batch_size, m, n);
  in_a_buffer.setRandom();
  in_b_buffer.setRandom();
  for (int i = 0; i < batch_size; ++i) {
    Eigen::Map<Matrix, Eigen::Unaligned> a(in_a_buffer.data() + i * m * k, m, k);
    Eigen::Map<Matrix, Eigen::Unaligned> b(in_b_buffer.data() + i * k * n, k, n);
    Eigen::Map<Matrix, Eigen::Unaligned> c(out_c_buffer.data() + i * m * n, m, n);
    c = a * b;
  }
  int64_t a_size = batch_size * m * k * sizeof(T);
  int64_t b_size = batch_size * k * n * sizeof(T);
  int64_t c_size = batch_size * m * n * sizeof(T);

  Eigen::array<int, 3> shuffling({0, 2, 1});
  Eigen::Tensor<T, 3, Eigen::RowMajor> in_a_transposed = in_a_buffer.shuffle(shuffling);
  Eigen::Tensor<T, 3, Eigen::RowMajor> in_b_transposed = in_b_buffer.shuffle(shuffling);

  for (const auto& device_type : device_types) {
    if (device_type == DeviceType::kCPU && data_type == DataType::kFloat16) {
      // CPU matmul not support float16
      continue;
    }
    auto device = registry->GetDevice(device_type, 0);
    ep::test::PinnedMemoryGuard input_a(device.get(), a_size);
    ep::test::PinnedMemoryGuard input_b(device.get(), b_size);
    if (transpose_a) {
      std::memcpy(input_a.ptr(), in_a_transposed.data(), a_size);
    } else {
      std::memcpy(input_a.ptr(), in_a_buffer.data(), a_size);
    }
    if (transpose_b) {
      std::memcpy(input_b.ptr(), in_b_transposed.data(), b_size);
    } else {
      std::memcpy(input_b.ptr(), in_b_buffer.data(), b_size);
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
    std::unique_ptr<BatchMatmul> batch_matmul =
        NewPrimitive<BatchMatmulFactory>(device_type, data_type, trans_a, trans_b);
    ASSERT_TRUE(d2h.operator bool());
    ASSERT_TRUE(h2d.operator bool());
    ASSERT_TRUE(batch_matmul.operator bool());
    h2d->Launch(stream.stream(), device_a.ptr(), input_a.ptr(), a_size);
    h2d->Launch(stream.stream(), device_b.ptr(), input_b.ptr(), b_size);
    batch_matmul->Launch(stream.stream(), batch_size, m, n, k, 1.0, device_a.ptr(), device_b.ptr(),
                         0.0, device_c.ptr());
    d2h->Launch(stream.stream(), output.ptr(), device_c.ptr(), c_size);
    CHECK_JUST(stream.stream()->Sync());
    Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>, Eigen::Unaligned> eigen_out(
        out_c_buffer.data(), out_c_buffer.size());
    Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>, Eigen::Unaligned> of_out(
        reinterpret_cast<T*>(output.ptr()), out_c_buffer.size());
    ASSERT_TRUE(eigen_out.template isApprox(of_out, static_cast<T>(0.001)));
  }
}

template<DataType data_type, typename T>
void TestBatchMatmul(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types,
                     int batch_size, int m, int k, int n) {
  TestBatchMatmul<data_type, T>(registry, device_types, batch_size, m, k, n, false, false);
  TestBatchMatmul<data_type, T>(registry, device_types, batch_size, m, k, n, true, false);
  TestBatchMatmul<data_type, T>(registry, device_types, batch_size, m, k, n, false, true);
  TestBatchMatmul<data_type, T>(registry, device_types, batch_size, m, k, n, true, true);
}

template<DataType data_type, typename T>
void TestBatchMatmul(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types) {
  TestBatchMatmul<data_type, T>(registry, device_types, 10, 64, 16, 8);
  TestBatchMatmul<data_type, T>(registry, device_types, 12, 16, 7, 12);
}

}  // namespace

TEST_F(PrimitiveTest, TestBatchMatmul) {
  TestBatchMatmul<DataType::kDouble, double>(&device_manager_registry_, available_device_types_);
  TestBatchMatmul<DataType::kFloat, float>(&device_manager_registry_, available_device_types_);
  TestBatchMatmul<DataType::kFloat16, Eigen::half>(&device_manager_registry_,
                                                   available_device_types_);
}

}  // namespace test

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow
