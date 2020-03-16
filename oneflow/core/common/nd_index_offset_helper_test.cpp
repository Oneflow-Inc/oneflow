#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace test {

template<int ndims>
void test_3d() {
  const int64_t d0_max = 3;
  const int64_t d1_max = 4;
  const int64_t d2_max = 5;
  const NdIndexOffsetHelper<int64_t, ndims> helper(d0_max, d1_max, d2_max);
  for (int64_t d0 = 0; d0 < d0_max; ++d0) {
    const int64_t offset0 = d0 * d1_max * d2_max;
    {
      std::vector<int64_t> expected0({d0});
      {
        std::vector<int64_t> dims(1);
        helper.OffsetToNdIndex(offset0, dims.data(), 1);
        ASSERT_EQ(expected0, dims);
      }
      {
        std::vector<int64_t> dims(1);
        helper.OffsetToNdIndex(offset0, dims.at(0));
        ASSERT_EQ(expected0, dims);
      }
      ASSERT_EQ(offset0, helper.NdIndexToOffset(expected0.data(), 1));
      ASSERT_EQ(offset0, helper.NdIndexToOffset(expected0.at(0)));
    }
    for (int64_t d1 = 0; d1 < d1_max; ++d1) {
      const int64_t offset1 = offset0 + d1 * d2_max;
      {
        std::vector<int64_t> expected1({d0, d1});
        {
          std::vector<int64_t> dims(2);
          helper.OffsetToNdIndex(offset1, dims.data(), 2);
          ASSERT_EQ(expected1, dims);
        }
        {
          std::vector<int64_t> dims(2);
          helper.OffsetToNdIndex(offset1, dims.at(0), dims.at(1));
          ASSERT_EQ(expected1, dims);
        }
        ASSERT_EQ(offset1, helper.NdIndexToOffset(expected1.data(), 2));
        ASSERT_EQ(offset1, helper.NdIndexToOffset(expected1.at(0), expected1.at(1)));
      }
      for (int64_t d2 = 0; d2 < d2_max; ++d2) {
        const int64_t offset2 = offset1 + d2;
        {
          std::vector<int64_t> expected2({d0, d1, d2});
          {
            std::vector<int64_t> dims(3);
            helper.OffsetToNdIndex(offset2, dims.data(), 3);
            ASSERT_EQ(expected2, dims);
          }
          {
            std::vector<int64_t> dims(3);
            helper.OffsetToNdIndex(offset2, dims.at(0), dims.at(1), dims.at(2));
            ASSERT_EQ(expected2, dims);
          }
          if (ndims == 3) {
            std::vector<int64_t> dims(3);
            helper.OffsetToNdIndex(offset2, dims.data());
            ASSERT_EQ(expected2, dims);
            ASSERT_EQ(offset2, helper.NdIndexToOffset(expected2.data()));
          }
          ASSERT_EQ(offset2, helper.NdIndexToOffset(expected2.data(), 3));
          ASSERT_EQ(offset2,
                    helper.NdIndexToOffset(expected2.at(0), expected2.at(1), expected2.at(2)));
        }
      }
    }
  }
}

TEST(NdIndexOffsetHelper, static_3d) { test_3d<3>(); }

TEST(NdIndexOffsetHelper, dynamic_3d) {
  test_3d<4>();
  test_3d<8>();
}

}  // namespace test

}  // namespace oneflow
