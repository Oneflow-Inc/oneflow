#include "oneflow/core/ndarray/ndarray_helper.h"
#include <gtest/gtest.h>

namespace oneflow {

namespace test {

TEST(VarNdArray, one_elem_assign) {
  std::vector<int32_t> data({1});
  std::vector<int32_t> buffer({0});
  NdArrayHelper<int32_t, 1> ndarray;
  auto&& data_ndarray = ndarray.Var({1LL}, data.data());
  auto&& buffer_ndarray = ndarray.Var({1LL}, buffer.data());
  buffer_ndarray.Assign(data_ndarray);
  ASSERT_EQ(data[0], buffer[0]);
}

TEST(VarNdArray, 1d_assign) {
  std::vector<int32_t> data({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  std::vector<int32_t> buffer(10, 0);
  NdArrayHelper<int32_t, 1> ndarray;
  auto&& data_ndarray = ndarray.Var({10LL}, data.data());
  auto&& buffer_ndarray = ndarray.Var({10LL}, buffer.data());
  buffer_ndarray.Assign(data_ndarray);
  ASSERT_EQ(memcmp(data.data(), buffer.data(), sizeof(int32_t) * 10), 0);
}

}  // namespace test

}  // namespace oneflow
