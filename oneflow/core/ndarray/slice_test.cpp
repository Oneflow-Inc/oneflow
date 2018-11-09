#include "oneflow/core/ndarray/slice.h"
#include <gtest/gtest.h>

namespace oneflow {

namespace test {

TEST(Slice, size) {
  Slice slice({-2, 0, -1});
  slice.Bound(4);
  ASSERT_EQ(slice.Size(), 2);
}

TEST(Slice, contiguous) {
  Slice slice({0, -1, 1});
  slice.Bound(4);
  ASSERT_TRUE(slice.IsContiguous());
  ASSERT_FALSE(slice.IsCoveringAll());
}

TEST(Slice, is_covering_all) {
  Slice slice({});
  slice.Bound(4);
  ASSERT_TRUE(slice.IsCoveringAll());
  ASSERT_TRUE(slice.IsContiguous());
}

}  // namespace test

}  // namespace oneflow
