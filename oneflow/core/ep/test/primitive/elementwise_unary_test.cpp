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
#include "oneflow/core/ep/include/primitive/elementwise_unary.h"
#include <Eigen/Core>
namespace oneflow {

namespace ep {

namespace primitive {

namespace test {

template<typename Src, typename Dst>
struct ReluFunctor {
  Dst operator()(Src src) {
    if (src > zero_val) { return src; }
    return zero_val;
  }

  Src zero_val = static_cast<Src>(0.0);
};

template<typename Src, typename Dst>
struct GeluFunctor {
  Dst operator()(Src src) {
    return static_cast<Dst>(0.5) * src * (static_cast<Src>(1.0) + std::erf(inv_sqrt2 * src));
  }
  Src inv_sqrt2 = std::sqrt(0.5);
};

template<typename Src, typename Dst>
struct TanhFunctor {
  Dst operator()(Src src) { return static_cast<Dst>(std::tanh(src)); }
};

template<typename Src, typename Dst>
struct LogicalNotFunctor {
  Dst operator()(Src src) { return static_cast<Dst>(!src); }
};

template<typename Src, typename Dst, typename FunctorT>
void EigenElementwise(FunctorT functor, Src* src, Dst* dst, const size_t elem_cnt) {
  for (int idx = 0; idx < elem_cnt; idx++) { dst[idx] = functor(src[idx]); }
}

template<typename Src, typename Dst, DataType SrcType, DataType DstType,
         ep::primitive::UnaryOp unary_op, template<typename A, typename B> class FunctorClass>
void TestElementwise(DeviceManagerRegistry* registry, const std::set<DeviceType>& device_types,
                     const size_t elem_cnt, Scalar attr0 = Scalar(), Scalar attr1 = Scalar()) {
  for (const auto& device_type : device_types) {
    auto device = registry->GetDevice(device_type, 0);
    using EigenSrcVec = Eigen::Matrix<Src, 1, Eigen::Dynamic>;
    using EigenDstVec = Eigen::Matrix<Dst, 1, Eigen::Dynamic>;

    const size_t src_data_size = elem_cnt * sizeof(Src);
    const size_t dst_data_size = elem_cnt * sizeof(Dst);
    EigenSrcVec eigen_src(elem_cnt);
    EigenDstVec eigen_dst(elem_cnt);
    eigen_src.setRandom();
    eigen_dst.setZero();

    ep::test::PinnedMemoryGuard host_src(device.get(), elem_cnt * sizeof(Src));
    ep::test::PinnedMemoryGuard host_dst(device.get(), elem_cnt * sizeof(Dst));
    ep::test::DeviceMemoryGuard device_src(device.get(), elem_cnt * sizeof(Src));
    ep::test::DeviceMemoryGuard device_dst(device.get(), elem_cnt * sizeof(Dst));

    ep::test::StreamGuard stream(device.get());
    std::unique_ptr<ElementwiseUnary> elementwise_primitive = NewPrimitive<ElementwiseUnaryFactory>(
        device_type, unary_op, /*src_type=*/SrcType, /*dst_type=*/DstType, attr0, attr1);
    ASSERT_TRUE(elementwise_primitive.operator bool());
    std::unique_ptr<Memcpy> h2d = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kHtoD);
    std::unique_ptr<Memcpy> d2h = NewPrimitive<MemcpyFactory>(device_type, MemcpyKind::kDtoH);
    ASSERT_TRUE(d2h.operator bool());
    ASSERT_TRUE(h2d.operator bool());
    Src* eigen_src_data = eigen_src.data();
    std::memcpy(host_src.ptr(), eigen_src_data, src_data_size);
    h2d->Launch(stream.stream(), device_src.ptr<Src>(), host_src.ptr<Src>(), src_data_size);
    elementwise_primitive->Launch(stream.stream(), device_src.ptr<Dst>(), device_dst.ptr<Dst>(),
                                  elem_cnt);
    d2h->Launch(stream.stream(), host_dst.ptr<Dst>(), device_dst.ptr<Dst>(), dst_data_size);
    CHECK_JUST(stream.stream()->Sync());

    FunctorClass<Src, Dst> functor{};
    EigenElementwise<Src, Dst, FunctorClass<Src, Dst>>(functor, eigen_src.data(), eigen_dst.data(),
                                                       elem_cnt);
    auto elementwise_primitive_res =
        Eigen::Map<EigenDstVec, Eigen::Unaligned>(host_dst.ptr<Dst>(), elem_cnt);
    ASSERT_TRUE(eigen_dst.template isApprox(elementwise_primitive_res));
  }
}

TEST_F(PrimitiveTest, TestElementwisePrimitive) {
  // Test Relu
  TestElementwise<float, float, DataType::kFloat, DataType::kFloat, ep::primitive::UnaryOp::kRelu,
                  ReluFunctor>(&device_manager_registry_, available_device_types_, 16);
  TestElementwise<double, double, DataType::kDouble, DataType::kDouble,
                  ep::primitive::UnaryOp::kRelu, ReluFunctor>(&device_manager_registry_,
                                                              available_device_types_, 32);
  TestElementwise<int32_t, int32_t, DataType::kInt32, DataType::kInt32,
                  ep::primitive::UnaryOp::kRelu, ReluFunctor>(&device_manager_registry_,
                                                              available_device_types_, 64);
  TestElementwise<int64_t, int64_t, DataType::kInt64, DataType::kInt64,
                  ep::primitive::UnaryOp::kRelu, ReluFunctor>(&device_manager_registry_,
                                                              available_device_types_, 128);

  // Test Gelu
  TestElementwise<float, float, DataType::kFloat, DataType::kFloat, ep::primitive::UnaryOp::kGelu,
                  GeluFunctor>(&device_manager_registry_, available_device_types_, 32);
  TestElementwise<double, double, DataType::kDouble, DataType::kDouble,
                  ep::primitive::UnaryOp::kGelu, GeluFunctor>(&device_manager_registry_,
                                                              available_device_types_, 128);

  // Test Tanh
  TestElementwise<float, float, DataType::kFloat, DataType::kFloat, ep::primitive::UnaryOp::kTanh,
                  TanhFunctor>(&device_manager_registry_, available_device_types_, 32);
  TestElementwise<double, double, DataType::kDouble, DataType::kDouble,
                  ep::primitive::UnaryOp::kTanh, TanhFunctor>(&device_manager_registry_,
                                                              available_device_types_, 128);

  // Test Logical Not
  TestElementwise<float, bool, DataType::kFloat, DataType::kBool,
                  ep::primitive::UnaryOp::kLogicalNot, LogicalNotFunctor>(
      &device_manager_registry_, available_device_types_, 32);
  TestElementwise<double, bool, DataType::kDouble, DataType::kBool,
                  ep::primitive::UnaryOp::kLogicalNot, LogicalNotFunctor>(
      &device_manager_registry_, available_device_types_, 64);
  TestElementwise<int8_t, bool, DataType::kInt8, DataType::kBool,
                  ep::primitive::UnaryOp::kLogicalNot, LogicalNotFunctor>(
      &device_manager_registry_, available_device_types_, 16);
  TestElementwise<int32_t, bool, DataType::kInt32, DataType::kBool,
                  ep::primitive::UnaryOp::kLogicalNot, LogicalNotFunctor>(
      &device_manager_registry_, available_device_types_, 128);
  TestElementwise<int64_t, bool, DataType::kInt64, DataType::kBool,
                  ep::primitive::UnaryOp::kLogicalNot, LogicalNotFunctor>(
      &device_manager_registry_, available_device_types_, 96);
}

}  // namespace test

}  // namespace primitive

}  // namespace ep

}  // namespace oneflow
