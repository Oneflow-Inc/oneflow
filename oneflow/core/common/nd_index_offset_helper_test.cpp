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
// include sstream first to avoid some compiling error
// caused by the following trick
// reference: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=65899
#include <sstream>
#include "gtest/gtest.h"
#define private public
#define protected public
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace test {

template<typename T, int ndims>
void test_3d() {
  const T d0_max = 3;
  const T d1_max = 4;
  const T d2_max = 5;
  const NdIndexOffsetHelper<T, ndims> helper(d0_max, d1_max, d2_max);
  for (T d0 = 0; d0 < d0_max; ++d0) {
    const T offset0 = d0 * d1_max * d2_max;
    {
      std::vector<T> expected0({d0});
      {
        std::vector<T> dims(1);
        helper.OffsetToNdIndex(offset0, dims.data(), 1);
        ASSERT_EQ(expected0, dims);
      }
      {
        std::vector<T> dims(1);
        helper.OffsetToNdIndex(offset0, dims.at(0));
        ASSERT_EQ(expected0, dims);
      }
      ASSERT_EQ(offset0, helper.NdIndexToOffset(expected0.data(), 1));
      ASSERT_EQ(offset0, helper.NdIndexToOffset(expected0.at(0)));
    }
    for (T d1 = 0; d1 < d1_max; ++d1) {
      const T offset1 = offset0 + d1 * d2_max;
      {
        std::vector<T> expected1({d0, d1});
        {
          std::vector<T> dims(2);
          helper.OffsetToNdIndex(offset1, dims.data(), 2);
          ASSERT_EQ(expected1, dims);
        }
        {
          std::vector<T> dims(2);
          helper.OffsetToNdIndex(offset1, dims.at(0), dims.at(1));
          ASSERT_EQ(expected1, dims);
        }
        ASSERT_EQ(offset1, helper.NdIndexToOffset(expected1.data(), 2));
        ASSERT_EQ(offset1, helper.NdIndexToOffset(expected1.at(0), expected1.at(1)));
      }
      for (T d2 = 0; d2 < d2_max; ++d2) {
        const T offset2 = offset1 + d2;
        {
          std::vector<T> expected2({d0, d1, d2});
          {
            std::vector<T> dims(3);
            helper.OffsetToNdIndex(offset2, dims.data(), 3);
            ASSERT_EQ(expected2, dims);
          }
          {
            std::vector<T> dims(3);
            helper.OffsetToNdIndex(offset2, dims.at(0), dims.at(1), dims.at(2));
            ASSERT_EQ(expected2, dims);
          }
          if (ndims == 3) {
            std::vector<T> dims(3);
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

TEST(NdIndexOffsetHelper, static_3d) {
  test_3d<int32_t, 3>();
  test_3d<int64_t, 3>();
}

TEST(NdIndexOffsetHelper, dynamic_3d) {
  test_3d<int32_t, 4>();
  test_3d<int64_t, 4>();
  test_3d<int32_t, 8>();
  test_3d<int64_t, 8>();
}

template<typename T>
void test_constructor() {
  const T d0 = 3;
  const T d1 = 4;
  const T d2 = 5;
  // static
  {
    std::vector<T> dims({d0, d1, d2});
    const NdIndexOffsetHelper<T, 3> helper1(d0, d1, d2);
    const NdIndexOffsetHelper<T, 3> helper2(dims.data());
    const NdIndexOffsetHelper<T, 3> helper3(dims.data(), dims.size());
    std::vector<T> stride({d1 * d2, d2, 1});
    for (int i = 0; i < 3; ++i) {
      ASSERT_EQ(helper1.stride_[i], stride[i]);
      ASSERT_EQ(helper2.stride_[i], stride[i]);
      ASSERT_EQ(helper3.stride_[i], stride[i]);
    }
  }
  // dynamic
  {
    std::vector<T> dims({d0, d1, d2});
    const NdIndexOffsetHelper<T, 6> helper1(d0, d1, d2);
    const NdIndexOffsetHelper<T, 6> helper2(dims.data(), dims.size());
    std::vector<T> stride({d1 * d2, d2, 1, 1, 1, 1});
    for (int i = 0; i < 6; ++i) {
      ASSERT_EQ(helper1.stride_[i], stride[i]);
      ASSERT_EQ(helper2.stride_[i], stride[i]);
    }
  }
}

TEST(NdIndexOffsetHelper, constructor) {
  test_constructor<int32_t>();
  test_constructor<int64_t>();
}

template<typename T, typename U>
void test_stride_constructor() {
  const T d1 = 5;
  const T d2 = 6;

  const U u1 = 5;
  const U u2 = 6;

  std::vector<T> strides({d1 * d2, d2, 1});
  std::vector<U> strides_u({u1 * u2, u2, 1});

  const NdIndexStrideOffsetHelper<T, 3> helper1(strides.data());
  const NdIndexStrideOffsetHelper<T, 3> helper2(strides.data(), strides.size());
  const NdIndexStrideOffsetHelper<T, 3> helper3(strides_u.data());
  const NdIndexStrideOffsetHelper<T, 3> helper4(strides_u.data(), strides_u.size());

  for (int i = 0; i < 3; i++) {
    ASSERT_EQ(helper1.stride_[i], strides[i]);
    ASSERT_EQ(helper2.stride_[i], strides[i]);
    ASSERT_EQ(helper3.stride_[i], strides_u[i]);
    ASSERT_EQ(helper4.stride_[i], strides_u[i]);
  }
}

TEST(NdIndexStrideOffsetHelper, constructor) {
  test_stride_constructor<int32_t, int64_t>();
  test_stride_constructor<int64_t, int32_t>();
}

}  // namespace test

}  // namespace oneflow
