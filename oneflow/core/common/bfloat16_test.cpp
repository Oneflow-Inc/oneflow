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
#include "gtest/gtest.h"
#include "oneflow/core/common/bfloat16.h"
#include "oneflow/core/common/bfloat16_math.h"

namespace oneflow {
namespace test {

float float_from_bytes(uint32_t sign, uint32_t exponent, uint32_t fraction) {
  // reference: pytorch/c10/test/util/bfloat16_test.cpp
  // https://github.com/pytorch/pytorch/blob/release/1.12/c10/test/util/bfloat16_test.cpp
  uint32_t bytes = 0;
  bytes |= sign;
  bytes <<= 8;
  bytes |= exponent;
  bytes <<= 23;
  bytes |= fraction;
  float res = NAN;
  std::memcpy(&res, &bytes, sizeof(res));
  return res;
}

TEST(BFLOAT16MATH, Add) {
  // 6.25
  float input = float_from_bytes(0, 0, 0x40C80000U);
  // 7.25
  float expected = float_from_bytes(0, 0, 0x40E80000U);

  bfloat16 b(input);
  b = b + 1;

  float res = static_cast<float>(b);
  EXPECT_EQ(res, expected);
}

TEST(BFLOAT16MATH, Sub) {
  // 7.25
  float input = float_from_bytes(0, 0, 0x40E80000U);
  // 6.25
  float expected = float_from_bytes(0, 0, 0x40C80000U);

  bfloat16 b(input);
  b = b - 1;

  float res = static_cast<float>(b);
  EXPECT_EQ(res, expected);
}

TEST(BFLOAT16MATH, Mul) {
  // 3.125
  float input = float_from_bytes(0, 0, 0x40480000U);
  // 6.25
  float expected = float_from_bytes(0, 0, 0x40C80000U);

  bfloat16 b(input);
  b = b * 2;

  float res = static_cast<float>(b);
  EXPECT_EQ(res, expected);
}

TEST(BFLOAT16MATH, Div) {
  // 6.25
  float input = float_from_bytes(0, 0, 0x40C80000U);
  // 3.125
  float expected = float_from_bytes(0, 0, 0x40480000U);

  bfloat16 b(input);
  b = b / 2;

  float res = static_cast<float>(b);
  EXPECT_EQ(res, expected);
}

TEST(BFLOAT16MATH, Log2) {
  // 16
  float input = float_from_bytes(0, 0, 0x41800000U);
  // 4
  float expected = float_from_bytes(0, 0, 0x40800000U);

  bfloat16 b(input);
  b = std::log2(b);

  float res = static_cast<float>(b);
  EXPECT_EQ(res, expected);
}

TEST(BFLOAT16MATH, Log10) {
  // 100
  float input = float_from_bytes(0, 0, 0x42C80000U);
  // 2
  float expected = float_from_bytes(0, 0, 0x40000000U);

  bfloat16 b(input);
  b = std::log10(b);

  float res = static_cast<float>(b);
  EXPECT_EQ(res, expected);
}

TEST(BFLOAT16MATH, Sqrt) {
  // 25
  float input = float_from_bytes(0, 0, 0x41C80000U);
  // 5
  float expected = float_from_bytes(0, 0, 0x40A00000U);

  bfloat16 b(input);
  b = std::sqrt(b);

  float res = static_cast<float>(b);
  EXPECT_EQ(res, expected);
}

}  // namespace test
}  // namespace oneflow
