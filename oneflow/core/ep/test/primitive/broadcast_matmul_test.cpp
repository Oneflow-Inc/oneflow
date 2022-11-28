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
#include "oneflow/core/ep/include/primitive/broadcast_matmul.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace oneflow {

namespace ep {

namespace primitive {

namespace test {

namespace {

template<DataType data_type, typename T>
void TestBroadcastMatmul(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types,
                         int batch_size, int m, int k, int n, bool transpose_a, bool transpose_b,
                         bool broadcast_a, bool broadcast_b, bool reduce_c) {
  using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  CHECK((!broadcast_a) || (!broadcast_b));
  int a_batch_dims = broadcast_a ? 1 : batch_size;
  int b_batch_dims = broadcast_b ? 1 : batch_size;
  int c_batch_dims = reduce_c ? 1 : batch_size;
  Eigen::Tensor<T, 3, Eigen::RowMajor> in_a_buffer(a_batch_dims, m, k);
  Eigen::Tensor<T, 3, Eigen::RowMajor> in_b_buffer(b_batch_dims, k, n);
  Eigen::Tensor<T, 3, Eigen::RowMajor> out_c_buffer(c_batch_dims, m, n);
  Eigen::Tensor<T, 3, Eigen::RowMajor> broadcast_c_buffer(batch_size, m, n);
  in_a_buffer.setRandom();
  in_b_buffer.setRandom();
  for (int i = 0; i < batch_size; ++i) {
    int64_t a_offset = broadcast_a ? 0 : i * m * k;
    int64_t b_offset = broadcast_b ? 0 : i * k * n;
    Eigen::Map<Matrix, Eigen::Unaligned> a(in_a_buffer.data() + a_offset, m, k);
    Eigen::Map<Matrix, Eigen::Unaligned> b(in_b_buffer.data() + b_offset, k, n);
    Eigen::Map<Matrix, Eigen::Unaligned> c(broadcast_c_buffer.data() + i * m * n, m, n);
    c = a * b;
  }
  if (reduce_c) {
    Eigen::array<int, 1> reduce_dim = {0};
    out_c_buffer = broadcast_c_buffer.sum(reduce_dim).eval().reshape(out_c_buffer.dimensions());
  } else {
    out_c_buffer = broadcast_c_buffer;
  }
  int64_t a_size = a_batch_dims * m * k * sizeof(T);
  int64_t b_size = b_batch_dims * k * n * sizeof(T);
  int64_t c_size = c_batch_dims * m * n * sizeof(T);
  Eigen::array<int, 3> shuffling({0, 2, 1});
  Eigen::Tensor<T, 3, Eigen::RowMajor> in_a_transposed = in_a_buffer.shuffle(shuffling);
  Eigen::Tensor<T, 3, Eigen::RowMajor> in_b_transposed = in_b_buffer.shuffle(shuffling);

  size_t num_a_dims = broadcast_a ? 2 : 3;
  std::vector<int64_t> a_dims;
  if (!broadcast_a) { a_dims.push_back(batch_size); }
  if (transpose_a) {
    a_dims.push_back(k);
    a_dims.push_back(m);
  } else {
    a_dims.push_back(m);
    a_dims.push_back(k);
  }
  size_t num_b_dims = broadcast_b ? 2 : 3;
  std::vector<int64_t> b_dims;
  if (!broadcast_b) { b_dims.push_back(batch_size); }
  if (transpose_b) {
    b_dims.push_back(n);
    b_dims.push_back(k);
  } else {
    b_dims.push_back(k);
    b_dims.push_back(n);
  }
  size_t num_c_dims = reduce_c ? 2 : 3;
  std::vector<int64_t> c_dims;
  if (!reduce_c) { c_dims.push_back(batch_size); }
  c_dims.push_back(m);
  c_dims.push_back(n);

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
    std::unique_ptr<BroadcastMatmul> broadcast_matmul =
        NewPrimitive<BroadcastMatmulFactory>(device_type, data_type, trans_a, trans_b, 3);
    ASSERT_TRUE(d2h.operator bool());
    ASSERT_TRUE(h2d.operator bool());
    ASSERT_TRUE(broadcast_matmul.operator bool());
    h2d->Launch(stream.stream(), device_a.ptr(), input_a.ptr(), a_size);
    h2d->Launch(stream.stream(), device_b.ptr(), input_b.ptr(), b_size);
    broadcast_matmul->Launch(stream.stream(), 1.0, num_a_dims, a_dims.data(), device_a.ptr(),
                             num_b_dims, b_dims.data(), device_b.ptr(), 0.0, num_c_dims,
                             c_dims.data(), device_c.ptr());
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
void TestBroadcastMatmul(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types,
                         int m, int k, int n, bool transpose_a, bool transpose_b) {
  TestBroadcastMatmul<data_type, T>(registry, device_types, 10, m, k, n, transpose_a, transpose_b,
                                    false, false, true);
  TestBroadcastMatmul<data_type, T>(registry, device_types, 10, m, k, n, transpose_a, transpose_b,
                                    false, false, false);
  TestBroadcastMatmul<data_type, T>(registry, device_types, 10, m, k, n, transpose_a, transpose_b,
                                    false, true, true);
  TestBroadcastMatmul<data_type, T>(registry, device_types, 10, m, k, n, transpose_a, transpose_b,
                                    false, true, false);
  TestBroadcastMatmul<data_type, T>(registry, device_types, 12, m, k, n, transpose_a, transpose_b,
                                    true, false, true);
  TestBroadcastMatmul<data_type, T>(registry, device_types, 12, m, k, n, transpose_a, transpose_b,
                                    true, false, false);
}

template<DataType data_type, typename T>
void TestBroadcastMatmul(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types,
                         int m, int k, int n) {
  TestBroadcastMatmul<data_type, T>(registry, device_types, m, k, n, false, false);
  TestBroadcastMatmul<data_type, T>(registry, device_types, m, k, n, true, false);
  TestBroadcastMatmul<data_type, T>(registry, device_types, m, k, n, false, true);
  TestBroadcastMatmul<data_type, T>(registry, device_types, m, k, n, true, true);
}

template<DataType data_type, typename T>
void TestBroadcastMatmul(DeviceManagerRegistry* registry,
                         const std::set<DeviceType>& device_types) {
  TestBroadcastMatmul<data_type, T>(registry, device_types, 64, 16, 8);
  TestBroadcastMatmul<data_type, T>(registry, device_types, 16, 7, 12);
}

}  // namespace

TEST_F(PrimitiveTest, TestBroadcastMatmul) {
  TestBroadcastMatmul<DataType::kDouble, double>(&device_manager_registry_,
                                                 available_device_types_);
  TestBroadcastMatmul<DataType::kFloat, float>(&device_manager_registry_, available_device_types_);
  TestBroadcastMatmul<DataType::kFloat16, Eigen::half>(&device_manager_registry_,
                                                       available_device_types_);
}

}  // namespace test

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow
