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
#include "oneflow/core/ep/include/primitive/cast.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace oneflow {

namespace ep {

namespace primitive {

namespace test {

namespace {

template<DataType src_data_type, typename Src, DataType dst_data_type, typename Dst>
void TestCast(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types,
              int elem_cnt) {
  if (src_data_type == dst_data_type) { return; }
  if (dst_data_type == kFloat16 && src_data_type != kFloat) { return; }
  const int src_data_size = elem_cnt * sizeof(Src);
  const int dst_data_size = elem_cnt * sizeof(Dst);
  Eigen::Tensor<Src, 1, Eigen::RowMajor> cast_in(elem_cnt);
  Eigen::Tensor<Dst, 1, Eigen::RowMajor> cast_out(elem_cnt);
  cast_in.setRandom();
  cast_out = cast_in.template cast<Dst>();

  for (const auto& device_type : device_types) {
    auto device = registry->GetDevice(device_type, 0);
    ep::test::PinnedMemoryGuard input(device.get(), src_data_size);
    ep::test::PinnedMemoryGuard output(device.get(), dst_data_size);
    std::memcpy(input.ptr(), cast_in.data(), src_data_size);
    ep::test::DeviceMemoryGuard device_in(device.get(), src_data_size);
    ep::test::DeviceMemoryGuard device_out(device.get(), dst_data_size);
    ep::test::StreamGuard stream(device.get());
    std::unique_ptr<Memcpy> h2d = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kHtoD);
    ASSERT_TRUE(h2d.operator bool());
    std::unique_ptr<Memcpy> d2h = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kDtoH);
    ASSERT_TRUE(d2h.operator bool());
    h2d->Launch(stream.stream(), device_in.ptr(), input.ptr(), src_data_size);
    std::unique_ptr<Cast> cast =
        NewPrimitive<CastFactory>(device_type, src_data_type, dst_data_type);
    ASSERT_TRUE(cast.operator bool());
    cast->Launch(stream.stream(), device_in.ptr(), device_out.ptr(), elem_cnt);
    d2h->Launch(stream.stream(), output.ptr(), device_out.ptr(), dst_data_size);
    CHECK_JUST(stream.stream()->Sync());
    Eigen::Map<Eigen::Matrix<Dst, 1, Eigen::Dynamic>, Eigen::Unaligned> eigen_out(cast_out.data(),
                                                                                  cast_out.size());
    Eigen::Map<Eigen::Matrix<Dst, 1, Eigen::Dynamic>, Eigen::Unaligned> of_out(
        reinterpret_cast<Dst*>(output.ptr()), cast_out.size());
    ASSERT_TRUE(eigen_out.template isApprox(of_out));
  }
}

template<DataType src_data_type, typename Src>
void TestCast(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types,
              int elem_cnt) {
  TestCast<src_data_type, Src, DataType::kBool, bool>(registry, device_types, elem_cnt);
  TestCast<src_data_type, Src, DataType::kInt8, int8_t>(registry, device_types, elem_cnt);
  TestCast<src_data_type, Src, DataType::kUInt8, uint8_t>(registry, device_types, elem_cnt);
  TestCast<src_data_type, Src, DataType::kInt32, int32_t>(registry, device_types, elem_cnt);
  TestCast<src_data_type, Src, DataType::kInt64, int64_t>(registry, device_types, elem_cnt);
  TestCast<src_data_type, Src, DataType::kFloat, float>(registry, device_types, elem_cnt);
  TestCast<src_data_type, Src, DataType::kDouble, double>(registry, device_types, elem_cnt);
  TestCast<src_data_type, Src, DataType::kFloat16, Eigen::half>(registry, device_types, elem_cnt);
}

void TestCast(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types,
              int elem_cnt) {
  TestCast<DataType::kBool, bool>(registry, device_types, elem_cnt);
  TestCast<DataType::kInt8, int8_t>(registry, device_types, elem_cnt);
  TestCast<DataType::kUInt8, uint8_t>(registry, device_types, elem_cnt);
  TestCast<DataType::kInt32, int32_t>(registry, device_types, elem_cnt);
  TestCast<DataType::kInt64, int64_t>(registry, device_types, elem_cnt);
  TestCast<DataType::kFloat, float>(registry, device_types, elem_cnt);
  TestCast<DataType::kDouble, double>(registry, device_types, elem_cnt);
  TestCast<DataType::kFloat16, Eigen::half>(registry, device_types, elem_cnt);
}

}  // namespace

TEST_F(PrimitiveTest, TestCast) {
  std::vector<int> elem_cnts = {1024, 3193, 5765};
  for (int i = 0; i < elem_cnts.size(); ++i) {
    TestCast(&device_manager_registry_, available_device_types_, elem_cnts.at(i));
  }
}

}  // namespace test

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow
