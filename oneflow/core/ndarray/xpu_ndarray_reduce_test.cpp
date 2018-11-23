#include "oneflow/core/ndarray/xpu_ndarray_util.h"
#include <gtest/gtest.h>

namespace oneflow {

namespace test {

namespace {

void TestMiddleAxis(int num) {
  std::vector<int32_t> data(num * num * num, 1);
  std::vector<int32_t> tmp_storage(num * num * num, -8888);
  XpuVarNdarray<const int32_t> x(XpuShape(Shape({num, num, num})), data.data());
  XpuVarNdarray<int32_t> tmp(XpuShape(Shape({num, num, num})), tmp_storage.data());
  std::vector<int32_t> ret(num * num, -999);
  XpuVarNdarray<int32_t> y(XpuShape(Shape({num, 1, num})), ret.data());
  NdArrayReduce<DeviceType::kCPU, int32_t, 3>::Reduce(nullptr, y, x, tmp);
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < num; ++j) { ASSERT_EQ(ret[i * num + j], num); }
  }
}

}  // namespace

TEST(NdArrayReduce, sum) {
  std::vector<int32_t> data(100, 1);
  std::vector<int32_t> tmp_storage(100, -1);
  XpuVarNdarray<const int32_t> x(XpuShape(Shape({100})), data.data());
  XpuVarNdarray<int32_t> tmp(XpuShape(Shape({100})), tmp_storage.data());
  int32_t ret = -100;
  XpuVarNdarray<int32_t> y(XpuShape(Shape({1})), &ret);
  NdArrayReduce<DeviceType::kCPU, int32_t, 1>::Reduce(nullptr, y, x, tmp);
  ASSERT_EQ(ret, 100);
}

TEST(NdArrayReduce, middle_axis_2) { TestMiddleAxis(10); }

TEST(NdArrayReduce, middle_axis_10) { TestMiddleAxis(125); }

}  // namespace test

}  // namespace oneflow
