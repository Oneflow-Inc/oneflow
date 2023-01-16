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
#include "oneflow/core/ep/test/primitive/primitive_test.h"
#include "oneflow/core/ep/include/primitive/memset.h"
#include "oneflow/core/ep/include/primitive/memcpy.h"
#include "oneflow/core/ep/include/primitive/where.h"
#include "oneflow/core/common/data_type.h"

#include <gtest/gtest.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <sstream>

namespace oneflow {

template<>
struct GetDataType<Eigen::half> : std::integral_constant<DataType, DataType::kFloat16> {};

namespace ep {
namespace primitive {
namespace test {

namespace {

template<typename dims_type>
std::string DimsToString(const dims_type& dims, const std::string& name) {
  std::ostringstream ss;
  ss << name << "=(";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i > 0) { ss << ", "; }
    ss << dims[i];
  }
  ss << ")";
  return ss.str();
};

template<typename T, typename CondT, size_t ndim>
void TestWhere(const std::vector<Device*>& devices, size_t num_cond_dims, const int64_t* cond_dims,
               size_t num_x_dims, const int64_t* x_dims, size_t num_y_dims, const int64_t* y_dims) {
  ASSERT_TRUE(num_cond_dims <= ndim);
  ASSERT_TRUE(num_x_dims <= ndim);
  ASSERT_TRUE(num_y_dims <= ndim);

  std::array<int64_t, ndim> broadcast_dims{};
  std::array<int64_t, ndim> broadcast_cond_dims{};
  std::array<int64_t, ndim> broadcast_x_dims{};
  std::array<int64_t, ndim> broadcast_y_dims{};
  std::array<int64_t, ndim> extend_cond_dims{};
  std::array<int64_t, ndim> extend_x_dims{};
  std::array<int64_t, ndim> extend_y_dims{};
  for (size_t i = 0; i < ndim; ++i) {
    size_t cond_lpad = ndim - num_cond_dims;
    size_t x_lpad = ndim - num_x_dims;
    size_t y_lpad = ndim - num_y_dims;
    int64_t cond_dim = (i < cond_lpad) ? 1 : cond_dims[i - cond_lpad];
    int64_t x_dim = (i < x_lpad) ? 1 : x_dims[i - x_lpad];
    int64_t y_dim = (i < y_lpad) ? 1 : y_dims[i - y_lpad];
    int64_t max_dim = std::max(x_dim, y_dim);
    max_dim = std::max(max_dim, cond_dim);
    ASSERT_TRUE((cond_dim == 1 || cond_dim == max_dim) && (x_dim == 1 || x_dim == max_dim)
                && (y_dim == 1 || y_dim == max_dim));
    broadcast_dims[i] = max_dim;
    broadcast_cond_dims[i] = (cond_dim == max_dim) ? 1 : max_dim;
    broadcast_x_dims[i] = (x_dim == max_dim) ? 1 : max_dim;
    broadcast_y_dims[i] = (y_dim == max_dim) ? 1 : max_dim;
    extend_cond_dims[i] = cond_dim;
    extend_x_dims[i] = x_dim;
    extend_y_dims[i] = y_dim;
  }

  size_t cond_size = std::accumulate(extend_cond_dims.begin(), extend_cond_dims.end(), 1,
                                     std::multiplies<int64_t>());
  size_t x_size =
      std::accumulate(extend_x_dims.begin(), extend_x_dims.end(), 1, std::multiplies<int64_t>());
  size_t y_size =
      std::accumulate(extend_y_dims.begin(), extend_y_dims.end(), 1, std::multiplies<int64_t>());
  size_t z_size =
      std::accumulate(broadcast_dims.begin(), broadcast_dims.end(), 1, std::multiplies<int64_t>());
  size_t cond_byte_size = cond_size * sizeof(CondT);
  size_t x_byte_size = x_size * sizeof(T);
  size_t y_byte_size = y_size * sizeof(T);
  size_t z_byte_size = z_size * sizeof(T);

  // Eigen contrast
  Eigen::Tensor<T, ndim, Eigen::RowMajor> tensor_c(extend_cond_dims);
  Eigen::Tensor<T, ndim, Eigen::RowMajor> tensor_x(extend_x_dims);
  Eigen::Tensor<T, ndim, Eigen::RowMajor> tensor_y(extend_y_dims);
  tensor_c.setRandom();
  tensor_x.setRandom();
  tensor_y.setRandom();
  tensor_c = tensor_c.unaryExpr([](T x) -> T { return x > T{0} ? T{1} : T{0}; });
  Eigen::Tensor<CondT, ndim, Eigen::RowMajor> tensor_cond = tensor_c.template cast<CondT>();
  auto broadcast_c = tensor_cond.broadcast(broadcast_cond_dims);
  auto broadcast_x = tensor_x.broadcast(broadcast_x_dims);
  auto broadcast_y = tensor_y.broadcast(broadcast_y_dims);
  Eigen::Tensor<T, ndim, Eigen::RowMajor> tensor_z = broadcast_c.select(broadcast_x, broadcast_y);
  ASSERT_TRUE(tensor_z.size() == z_size) << tensor_z.size() << " vs. " << z_size << ", ";

  // test on devices
  for (auto* device : devices) {
    if (device->device_type() == DeviceType::kCPU && GetDataType<T>() == DataType::kFloat16) {
      // CPU matmul not support float16
      continue;
    }

    ep::test::PinnedMemoryGuard host_cond(device, cond_byte_size);
    ep::test::PinnedMemoryGuard host_x(device, x_byte_size);
    ep::test::PinnedMemoryGuard host_y(device, y_byte_size);
    ep::test::DeviceMemoryGuard cond(device, cond_byte_size);
    ep::test::DeviceMemoryGuard x(device, x_byte_size);
    ep::test::DeviceMemoryGuard y(device, y_byte_size);
    ep::test::DeviceMemoryGuard z(device, z_byte_size);
    ep::test::PinnedMemoryGuard host_z(device, z_byte_size);

    std::memcpy(host_cond.ptr(), tensor_cond.data(), cond_byte_size);
    std::memcpy(host_x.ptr(), tensor_x.data(), x_byte_size);
    std::memcpy(host_y.ptr(), tensor_y.data(), y_byte_size);

    ep::test::StreamGuard stream(device);
    auto h2d = NewPrimitive<MemcpyFactory>(device->device_type(), MemcpyKind::kHtoD);
    auto d2h = NewPrimitive<MemcpyFactory>(device->device_type(), MemcpyKind::kDtoH);
    auto where = NewPrimitive<WhereFactory>(device->device_type());
    ASSERT_TRUE(d2h.operator bool());
    ASSERT_TRUE(h2d.operator bool());
    ASSERT_TRUE(where.operator bool());

    h2d->Launch(stream.stream(), cond.ptr(), host_cond.ptr(), cond_byte_size);
    h2d->Launch(stream.stream(), x.ptr(), host_x.ptr(), x_byte_size);
    h2d->Launch(stream.stream(), y.ptr(), host_y.ptr(), y_byte_size);
    where->Launch(stream.stream(), GetDataType<CondT>(), num_cond_dims, cond_dims, cond.ptr(),
                  GetDataType<T>(), num_x_dims, x_dims, x.ptr(), num_y_dims, y_dims, y.ptr(),
                  z.ptr());
    d2h->Launch(stream.stream(), host_z.ptr(), z.ptr(), z_byte_size);
    CHECK_JUST(stream.stream()->Sync());

    Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>, Eigen::Unaligned> eigen_out(tensor_z.data(),
                                                                                tensor_z.size());
    Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>, Eigen::Unaligned> of_out(
        reinterpret_cast<T*>(host_z.ptr()), z_size);
    ASSERT_TRUE(eigen_out.template isApprox(of_out, static_cast<T>(1e-5)));
  }
}

template<typename T, typename CondT, size_t ndim>
void TestWhere(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types,
               const std::vector<int64_t>& cond_dims, const std::vector<int64_t>& x_dims,
               const std::vector<int64_t>& y_dims) {
  std::vector<Device*> devices;
  for (const auto& device_type : device_types) {
    auto device = registry->GetDevice(device_type, 0);
    ASSERT_TRUE(device);
    devices.push_back(device.get());
  }
  TestWhere<T, CondT, ndim>(devices, cond_dims.size(), cond_dims.data(), x_dims.size(),
                            x_dims.data(), y_dims.size(), y_dims.data());
}

}  // namespace

TEST_F(PrimitiveTest, TestWhere) {
  test::TestWhere<float, bool, 2>(&device_manager_registry_, available_device_types_, {4, 8},
                                  {4, 8}, {4, 8});
  test::TestWhere<bool, bool, 2>(&device_manager_registry_, available_device_types_, {4, 1}, {1, 8},
                                 {1, 8});
  test::TestWhere<uint8_t, bool, 2>(&device_manager_registry_, available_device_types_, {4, 1},
                                    {1, 8}, {1, 8});
  test::TestWhere<int32_t, bool, 2>(&device_manager_registry_, available_device_types_, {4, 1},
                                    {1, 8}, {1, 8});
  test::TestWhere<Eigen::half, bool, 2>(&device_manager_registry_, available_device_types_, {4, 1},
                                        {1, 8}, {1, 8});
  test::TestWhere<double, bool, 2>(&device_manager_registry_, available_device_types_, {4, 1},
                                   {1, 8}, {1, 8});
  test::TestWhere<bool, int32_t, 2>(&device_manager_registry_, available_device_types_, {1, 8},
                                    {4, 8}, {1});
  test::TestWhere<uint8_t, int32_t, 2>(&device_manager_registry_, available_device_types_, {1, 8},
                                       {4, 8}, {1});
  test::TestWhere<int32_t, int32_t, 2>(&device_manager_registry_, available_device_types_, {1, 8},
                                       {4, 8}, {1});
  test::TestWhere<float, int32_t, 2>(&device_manager_registry_, available_device_types_, {1, 8},
                                     {4, 8}, {1});
  test::TestWhere<Eigen::half, int32_t, 2>(&device_manager_registry_, available_device_types_,
                                           {1, 8}, {4, 8}, {1});
  test::TestWhere<double, int32_t, 2>(&device_manager_registry_, available_device_types_, {1, 8},
                                      {4, 8}, {1});
}

}  // namespace test
}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
