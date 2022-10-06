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
#include "oneflow/core/ep/include/primitive/softmax.h"
#include "oneflow/core/ep/include/primitive/log_softmax.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace oneflow {

namespace ep {

namespace primitive {

namespace test {

namespace {

template<DataType data_type, typename T>
void TestSoftmax(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types,
                 int num_rows, int num_cols, bool log_softmax) {
  const int elem_cnt = num_rows * num_cols;
  const int data_size = elem_cnt * sizeof(T);
  Eigen::Tensor<T, 2, Eigen::RowMajor> softmax_in(num_rows, num_cols);
  Eigen::Tensor<T, 2, Eigen::RowMajor> softmax_out(num_rows, num_cols);
  softmax_in.setRandom();
  Eigen::array<int, 1> reduce_dim = {1};
  Eigen::array<int, 2> reduced_shape = {num_rows, 1};
  Eigen::array<int, 2> broadcast_shape = {1, num_cols};

  Eigen::Tensor<T, 2, Eigen::RowMajor> row_buf =
      (softmax_in
       - softmax_in.maximum(reduce_dim).eval().reshape(reduced_shape).broadcast(broadcast_shape));
  if (log_softmax) {
    softmax_out = row_buf
                  - row_buf.exp()
                        .sum(reduce_dim)
                        .eval()
                        .reshape(reduced_shape)
                        .log()
                        .broadcast(broadcast_shape);
  } else {
    row_buf = row_buf.exp();
    softmax_out =
        row_buf / row_buf.sum(reduce_dim).eval().reshape(reduced_shape).broadcast(broadcast_shape);
  }

  for (const auto& device_type : device_types) {
    if (device_type == DeviceType::kCPU && data_type == DataType::kFloat16) {
      // CPU softmax not support float16
      continue;
    }
    auto device = registry->GetDevice(device_type, 0);
    ep::test::PinnedMemoryGuard input(device.get(), data_size);
    ep::test::PinnedMemoryGuard output(device.get(), data_size);
    std::memcpy(input.ptr(), softmax_in.data(), data_size);
    ep::test::DeviceMemoryGuard device_in(device.get(), data_size);
    ep::test::DeviceMemoryGuard device_out(device.get(), data_size);
    ep::test::StreamGuard stream(device.get());
    std::unique_ptr<Memcpy> h2d = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kHtoD);
    ASSERT_TRUE(h2d.operator bool());
    std::unique_ptr<Memcpy> d2h = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kDtoH);
    ASSERT_TRUE(d2h.operator bool());
    h2d->Launch(stream.stream(), device_in.ptr(), input.ptr(), data_size);
    if (log_softmax) {
      std::unique_ptr<LogSoftmax> log_softmax =
          NewPrimitive<LogSoftmaxFactory>(device_type, data_type);
      ASSERT_TRUE(log_softmax.operator bool());
      log_softmax->Launch(stream.stream(), num_rows, num_cols, device_in.ptr(), device_out.ptr());
    } else {
      std::unique_ptr<Softmax> softmax = NewPrimitive<SoftmaxFactory>(device_type, data_type);
      ASSERT_TRUE(softmax.operator bool());
      softmax->Launch(stream.stream(), num_rows, num_cols, device_in.ptr(), device_out.ptr());
    }
    d2h->Launch(stream.stream(), output.ptr(), device_out.ptr(), data_size);
    CHECK_JUST(stream.stream()->Sync());
    Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>, Eigen::Unaligned> eigen_out(softmax_out.data(),
                                                                                softmax_out.size());
    Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>, Eigen::Unaligned> of_out(
        reinterpret_cast<T*>(output.ptr()), softmax_out.size());
    ASSERT_TRUE(eigen_out.template isApprox(of_out, static_cast<T>(0.001)));
  }
}

void TestSoftmax(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types,
                 int num_rows, int num_cols) {
  TestSoftmax<DataType::kFloat, float>(registry, device_types, num_rows, num_cols, true);
  TestSoftmax<DataType::kFloat, float>(registry, device_types, num_rows, num_cols, false);
  TestSoftmax<DataType::kDouble, double>(registry, device_types, num_rows, num_cols, true);
  TestSoftmax<DataType::kDouble, double>(registry, device_types, num_rows, num_cols, false);
  TestSoftmax<DataType::kFloat16, Eigen::half>(registry, device_types, num_rows, num_cols, true);
  TestSoftmax<DataType::kFloat16, Eigen::half>(registry, device_types, num_rows, num_cols, false);
}

}  // namespace

TEST_F(PrimitiveTest, TestSoftmax) {
  std::vector<int> num_rows = {32, 33, 512, 511};
  std::vector<int> num_cols = {15, 16, 32, 768, 1536};
  for (int i = 0; i < num_rows.size(); ++i) {
    for (int j = 0; j < num_cols.size(); ++j) {
      TestSoftmax(&device_manager_registry_, available_device_types_, num_rows.at(i),
                  num_cols.at(j));
    }
  }
}

}  // namespace test

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow
