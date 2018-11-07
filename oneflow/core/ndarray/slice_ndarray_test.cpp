#include "oneflow/core/ndarray/ndarray_helper.h"
#include <gtest/gtest.h>

namespace oneflow {

namespace test {

TEST(SliceNdArray, one_elem_assign) {
  std::vector<int32_t> data({1});
  std::vector<int32_t> buffer({0});
  NdArrayHelper<int32_t, 1> ndarray;
  auto&& data_ndarray = ndarray.Var({1LL}, data.data());
  auto&& buffer_ndarray = ndarray.Var({1LL}, buffer.data());
  buffer_ndarray(0).Assign(data_ndarray(0));
  ASSERT_EQ(data[0], buffer[0]);
}

TEST(SliceNdArray, one_elem_assign_slice_on_slice) {
  std::vector<int32_t> data({1});
  std::vector<int32_t> buffer({0});
  NdArrayHelper<int32_t, 1> ndarray;
  auto&& data_ndarray = ndarray.Var({1LL}, data.data());
  auto&& buffer_ndarray = ndarray.Var({1LL}, buffer.data());
  buffer_ndarray(0)(0).Assign(data_ndarray(0)(0));
  ASSERT_EQ(data[0], buffer[0]);
}

TEST(SliceNdArray, 1d_assign) {
  std::vector<int32_t> data({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  std::vector<int32_t> buffer(10, 0);
  NdArrayHelper<int32_t, 1> ndarray;
  auto&& data_ndarray = ndarray.Var({10LL}, data.data());
  auto&& buffer_ndarray = ndarray.Var({10LL}, buffer.data());
  buffer_ndarray({}).Assign(data_ndarray({}));
  ASSERT_EQ(memcmp(data.data(), buffer.data(), sizeof(int32_t) * 10), 0);
}

TEST(SliceNdArray, 1d_slice_assign) {
  std::vector<int32_t> data({1, 2, 3, 4, 5, 6, 7, 8});
  std::vector<int32_t> buffer(10, 100);
  std::vector<int32_t> expected({100, 1, 2, 3, 4, 5, 6, 7, 8, 100});
  NdArrayHelper<int32_t, 1> ndarray;
  auto&& data_ndarray = ndarray.Var({static_cast<int64_t>(data.size())}, data.data());
  auto&& buffer_ndarray = ndarray.Var({10LL}, buffer.data());
  ASSERT_EQ(buffer_ndarray({1, -1}).shape(), Shape({8}));
  buffer_ndarray({1, -1}).Assign(data_ndarray({}));
  ASSERT_EQ(memcmp(expected.data(), buffer.data(), sizeof(int32_t) * 10), 0);
}

TEST(SliceNdArray, 1d_slice) {
  std::vector<int32_t> data({100, 1, 2, 3, 4, 5, 6, 7, 8, 100});
  std::vector<int32_t> buffer(8, 100);
  std::vector<int32_t> expected({1, 2, 3, 4, 5, 6, 7, 8});
  NdArrayHelper<int32_t, 1> ndarray;
  auto&& data_ndarray = ndarray.Var({static_cast<int64_t>(data.size())}, data.data());
  auto&& buffer_ndarray = ndarray.Var({static_cast<int64_t>(buffer.size())}, buffer.data());
  buffer_ndarray({}).Assign(data_ndarray({1, -1}));
  ASSERT_EQ(memcmp(expected.data(), buffer.data(), sizeof(int32_t) * buffer.size()), 0);
}

TEST(SliceNdArray, 2d_slice) {
  // clang-format off
  std::vector<int32_t> data({
      100, 100, 100, 100,
      100, 0,   1,   100,
      100, 2,   3,   100,
      100, 100, 100, 100,
  });
  // clang-format on
  std::vector<int32_t> buffer(4, 100);
  std::vector<int32_t> expected({0, 1, 2, 3});
  NdArrayHelper<int32_t, 2> ndarray;
  auto&& data_ndarray = ndarray.Var({4LL, 4LL}, data.data());
  auto&& buffer_ndarray = ndarray.Var({2LL, 2LL}, buffer.data());
  buffer_ndarray({}, {}).Assign(data_ndarray({1, -1}, {1, -1}));
  ASSERT_EQ(memcmp(expected.data(), buffer.data(), sizeof(int32_t) * buffer.size()), 0);
}

TEST(SliceNdArray, 2d_slice_assign) {
  std::vector<int32_t> data({0, 1, 2, 3});
  std::vector<int32_t> buffer(16, 100);
  // clang-format off
  std::vector<int32_t> expected({
      100, 100, 100, 100,
      100, 0,   1,   100,
      100, 2,   3,   100,
      100, 100, 100, 100,
  });
  // clang-format on
  NdArrayHelper<int32_t, 2> ndarray;
  auto&& data_ndarray = ndarray.Var({2LL, 2LL}, data.data());
  auto&& buffer_ndarray = ndarray.Var({4LL, 4LL}, buffer.data());
  buffer_ndarray({1, -1}, {1, -1}).Assign(data_ndarray({}, {}));
  ASSERT_EQ(memcmp(expected.data(), buffer.data(), sizeof(int32_t) * buffer.size()), 0);
}

TEST(SliceNdArray, 2d_slice_reverse) {
  // clang-format off
  std::vector<int32_t> data({
      100, 100, 100, 100,
      100, 0,   1,   100,
      100, 2,   3,   100,
      100, 100, 100, 100,
  });
  std::vector<int32_t> buffer(16, 100);
  std::vector<int32_t> expected({
      100, 100, 100, 100,
      100, 2,   3,   100,
      100, 0,   1,   100,
      100, 100, 100, 100,
  });
  // clang-format on
  NdArrayHelper<int32_t, 2> ndarray;
  auto&& data_ndarray = ndarray.Var({4LL, 4LL}, data.data());
  auto&& buffer_ndarray = ndarray.Var({4LL, 4LL}, buffer.data());
  buffer_ndarray({1, -1}, {1, -1}).Assign(data_ndarray({-2, 0, -1}, {1, -1}));
  ASSERT_EQ(memcmp(expected.data(), buffer.data(), sizeof(int32_t) * buffer.size()), 0);
}

}  // namespace test

}  // namespace oneflow
