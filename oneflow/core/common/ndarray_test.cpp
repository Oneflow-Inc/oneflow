#include "oneflow/core/common/ndarray.h"
#include <gtest/gtest.h>

namespace oneflow {

namespace test {

TEST(VarNdArray, 1d_assign) {
  std::vector<int32_t> data({1});
  std::vector<int32_t> buffer({0});
  VarNdArray<int32_t, 1> data_ndarray({1LL}, data.data());
  VarNdArray<int32_t, 1> buffer_ndarray({1LL}, buffer.data());
  data_ndarray.Assign(buffer_ndarray);
  ASSERT_EQ(data[0], buffer[0]);
}

TEST(SliceNdArray, 1d_assign) {
  std::vector<int32_t> data({1});
  std::vector<int32_t> buffer({0});
  VarNdArray<int32_t, 1> data_ndarray({1LL}, data.data());
  VarNdArray<int32_t, 1> buffer_ndarray({1LL}, buffer.data());
  data_ndarray(0).Assign(buffer_ndarray(0));
  ASSERT_EQ(data[0], buffer[0]);
}

}  // namespace test

}  // namespace oneflow
