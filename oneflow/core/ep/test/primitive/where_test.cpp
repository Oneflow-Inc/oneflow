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
    auto where = NewPrimitive<WhereFactory>(device->device_type(), GetDataType<CondT>(),
                                            GetDataType<T>(), ndim);
    ASSERT_TRUE(d2h.operator bool());
    ASSERT_TRUE(h2d.operator bool());
    ASSERT_TRUE(where.operator bool());

    h2d->Launch(stream.stream(), cond.ptr(), host_cond.ptr(), cond_byte_size);
    h2d->Launch(stream.stream(), x.ptr(), host_x.ptr(), x_byte_size);
    h2d->Launch(stream.stream(), y.ptr(), host_y.ptr(), y_byte_size);
    where->Launch(stream.stream(), num_cond_dims, cond_dims, cond.ptr(), num_x_dims, x_dims,
                  x.ptr(), num_y_dims, y_dims, y.ptr(), z.ptr());
    d2h->Launch(stream.stream(), host_z.ptr(), z.ptr(), z_byte_size);
    CHECK_JUST(stream.stream()->Sync());

    Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>, Eigen::Unaligned> eigen_out(tensor_z.data(),
                                                                                tensor_z.size());
    Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>, Eigen::Unaligned> of_out(
        reinterpret_cast<T*>(host_z.ptr()), z_size);
    ASSERT_TRUE(eigen_out.template isApprox(of_out));
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

template<typename T, typename = void>
struct random {};

template<>
struct random<bool, void> {
  bool operator()() {
    static std::default_random_engine e;
    static std::uniform_int_distribution<> dis(0, 1);
    return static_cast<bool>(dis(e));
  }
};

template<typename T>
struct random<T, std::enable_if_t<std::is_integral<T>::value>> {
  T operator()() {
    static std::default_random_engine e;
    static std::normal_distribution<> dis(0, 2);
    return dis(e);
  }
};

template<>
struct random<Eigen::half, void> {
  Eigen::half operator()() {
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(-1, 1);
    return Eigen::half{dis(e)};
  }
};

template<typename T>
struct random<T, std::enable_if_t<std::is_floating_point<T>::value>> {
  T operator()() {
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(-1, 1);
    return dis(e);
  }
};

template<typename T, typename CondT>
void TestScalarWhere(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types) {
  std::vector<Device*> devices;
  for (const auto& device_type : device_types) {
    auto device_ptr = registry->GetDevice(device_type, 0);
    ASSERT_TRUE(device_ptr);
    Device* device = device_ptr.get();

    CondT cond = random<bool>()();
    T x = random<T>()();
    T y = random<T>()();
    T z = cond ? x : y;

    ep::test::PinnedMemoryGuard host_cond(device, sizeof(CondT));
    ep::test::PinnedMemoryGuard host_x(device, sizeof(T));
    ep::test::PinnedMemoryGuard host_y(device, sizeof(T));
    ep::test::DeviceMemoryGuard device_cond(device, sizeof(CondT));
    ep::test::DeviceMemoryGuard device_x(device, sizeof(T));
    ep::test::DeviceMemoryGuard device_y(device, sizeof(T));
    ep::test::DeviceMemoryGuard device_z(device, sizeof(T));
    ep::test::PinnedMemoryGuard host_z(device, sizeof(T));

    std::memcpy(host_cond.ptr(), &cond, sizeof(CondT));
    std::memcpy(host_x.ptr(), &x, sizeof(T));
    std::memcpy(host_y.ptr(), &y, sizeof(T));

    ep::test::StreamGuard stream(device);
    auto h2d = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kHtoD);
    auto d2h = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kDtoH);
    auto where = NewPrimitive<WhereFactory>(device_type, GetDataType<CondT>(), GetDataType<T>(), 0);
    ASSERT_TRUE(d2h.operator bool());
    ASSERT_TRUE(h2d.operator bool());
    ASSERT_TRUE(where.operator bool());

    h2d->Launch(stream.stream(), device_cond.ptr(), host_cond.ptr(), sizeof(CondT));
    h2d->Launch(stream.stream(), device_x.ptr(), host_x.ptr(), sizeof(T));
    h2d->Launch(stream.stream(), device_y.ptr(), host_y.ptr(), sizeof(T));
    where->Launch(stream.stream(), 0, nullptr, device_cond.ptr(), 0, nullptr, device_x.ptr(), 0,
                  nullptr, device_y.ptr(), device_z.ptr());
    d2h->Launch(stream.stream(), host_z.ptr(), device_z.ptr(), sizeof(T));
    CHECK_JUST(stream.stream()->Sync());

    ASSERT_TRUE(*host_z.ptr<T>() == z);
  }
}

}  // namespace

TEST_F(PrimitiveTest, TestWhere) {
  TestWhere<float, bool, 2>(&device_manager_registry_, available_device_types_, {4, 8}, {4, 8},
                            {4, 8});
  TestWhere<bool, bool, 2>(&device_manager_registry_, available_device_types_, {4, 1}, {1, 8},
                           {1, 8});
  TestWhere<uint8_t, bool, 2>(&device_manager_registry_, available_device_types_, {4, 1}, {1, 8},
                              {1, 8});
  TestWhere<int32_t, bool, 2>(&device_manager_registry_, available_device_types_, {4, 1}, {1, 8},
                              {1, 8});
  TestWhere<Eigen::half, bool, 2>(&device_manager_registry_, available_device_types_, {4, 1},
                                  {1, 8}, {1, 8});
  TestWhere<double, bool, 2>(&device_manager_registry_, available_device_types_, {4, 1}, {1, 8},
                             {1, 8});
  TestWhere<bool, int32_t, 2>(&device_manager_registry_, available_device_types_, {1, 8}, {4, 8},
                              {1});
  TestWhere<int32_t, int32_t, 2>(&device_manager_registry_, available_device_types_, {1, 8}, {4, 8},
                                 {1});
  TestWhere<float, int32_t, 2>(&device_manager_registry_, available_device_types_, {1, 8}, {4, 8},
                               {1});
  TestWhere<Eigen::half, int32_t, 2>(&device_manager_registry_, available_device_types_, {1, 8},
                                     {4, 8}, {1});
  TestWhere<double, int32_t, 2>(&device_manager_registry_, available_device_types_, {1, 8}, {4, 8},
                                {1});
  TestWhere<float, bool, 2>(&device_manager_registry_, available_device_types_, {1, 6}, {2, 6},
                            {2, 1});
  TestWhere<float, bool, 2>(&device_manager_registry_, available_device_types_, {3, 7}, {3, 1},
                            {1, 7});
  TestWhere<float, bool, 3>(&device_manager_registry_, available_device_types_, {1, 4, 8},
                            {4, 1, 8}, {1, 1, 8});
  TestWhere<float, bool, 3>(&device_manager_registry_, available_device_types_, {1, 4, 8},
                            {4, 4, 8}, {1});
  TestWhere<float, bool, 4>(&device_manager_registry_, available_device_types_, {2, 1, 4, 8},
                            {1, 3, 4, 1}, {4, 8});
  TestScalarWhere<bool, bool>(&device_manager_registry_, available_device_types_);
  TestScalarWhere<float, bool>(&device_manager_registry_, available_device_types_);
  TestScalarWhere<Eigen::half, bool>(&device_manager_registry_, available_device_types_);
  TestScalarWhere<double, bool>(&device_manager_registry_, available_device_types_);
  TestScalarWhere<int32_t, bool>(&device_manager_registry_, available_device_types_);
  TestScalarWhere<bool, int32_t>(&device_manager_registry_, available_device_types_);
  TestScalarWhere<float, int32_t>(&device_manager_registry_, available_device_types_);
  TestScalarWhere<Eigen::half, int32_t>(&device_manager_registry_, available_device_types_);
  TestScalarWhere<double, int32_t>(&device_manager_registry_, available_device_types_);
  TestScalarWhere<int32_t, int32_t>(&device_manager_registry_, available_device_types_);
}

}  // namespace test
}  // namespace primitive
}  // namespace ep

}  // namespace oneflow
