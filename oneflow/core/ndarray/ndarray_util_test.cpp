#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include <gtest/gtest.h>

namespace oneflow {

namespace test {

TEST(NdarrayUtil, CpuBroadcastAdd3D) {
  std::vector<float> a(1 * 20 * 11);
  std::vector<float> b(1 * 20 * 1);
  std::vector<float> o(1 * 20 * 11);

  NdarrayUtil<DeviceType::kCPU, float>::BroadcastAdd(
      nullptr, XpuVarNdarray<float>(Shape({1, 20, 11}), o.data()),
      XpuVarNdarray<const float>(Shape({1, 20, 11}), a.data()),
      XpuVarNdarray<const float>(Shape({1, 20, 1}), b.data()));
}

}  // namespace test

}  // namespace oneflow
