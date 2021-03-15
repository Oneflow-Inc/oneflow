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
#include "util.h"
#include "oneflow/core/common/data_type_converter.h"
#include "oneflow/core/common/data_type_converter_test_static.h"
#ifdef __CUDA_ARCH__
#include <cuda_runtime.h>
#else
#include <cmath>
#endif

namespace oneflow {

namespace {

// cpp17 std::clamp possible implementation
template<class T>
constexpr const T& clamp(const T& v, const T& lo, const T& hi) {
  return (v < lo) ? lo : (hi < v) ? hi : v;
}

}  // namespace

TEST(ClampTest, Clamp) {
  ASSERT_TRUE(Clamp<uint8_t>(0) == 0);
  ASSERT_TRUE(Clamp<uint8_t>(255) == 255);
  ASSERT_TRUE(Clamp<uint8_t>(100) == 100);
  ASSERT_TRUE(Clamp<uint8_t>(100.3) == 100);
  ASSERT_TRUE(Clamp<uint8_t>(256) == 255);
  ASSERT_TRUE(Clamp<uint8_t>(-4) == 0);
  ASSERT_TRUE(Clamp<uint8_t>(-4.0f) == 0);
  ASSERT_TRUE(Clamp<uint8_t>(1e+20f) == 255);
  ASSERT_TRUE(Clamp<uint8_t>(-1e+20f) == 0);
  ASSERT_TRUE(Clamp<uint8_t>(1e+200) == 255);
  ASSERT_TRUE(Clamp<uint8_t>(-1e+200) == 0);

  ASSERT_TRUE(Clamp<int8_t>(-4) == -4);
  ASSERT_TRUE(Clamp<int8_t>(-4.2) == -4);
  ASSERT_TRUE(Clamp<int8_t>(4.2) == 4);
  ASSERT_TRUE(Clamp<int8_t>(127) == 127);
  ASSERT_TRUE(Clamp<int8_t>(128) == 127);
  ASSERT_TRUE(Clamp<int8_t>(256) == 127);
  ASSERT_TRUE(Clamp<int8_t>(-128) == -128);
  ASSERT_TRUE(Clamp<int8_t>(-256) == -128);
  ASSERT_TRUE(Clamp<int8_t>(1e+20f) == 127);
  ASSERT_TRUE(Clamp<int8_t>(-1e+20f) == -128);
  ASSERT_TRUE(Clamp<int8_t>(1e+200) == 127);
  ASSERT_TRUE(Clamp<int8_t>(-1e+200) == -128);

  ASSERT_TRUE(Clamp<uint16_t>(0) == 0);
  ASSERT_TRUE(Clamp<uint16_t>(0xffff) == 0xffff);
  ASSERT_TRUE(Clamp<uint16_t>(100) == 100);
  ASSERT_TRUE(Clamp<uint16_t>(100.3) == 100);
  ASSERT_TRUE(Clamp<uint16_t>(0x10000) == 0xffff);
  ASSERT_TRUE(Clamp<uint16_t>(-4) == 0);
  ASSERT_TRUE(Clamp<uint16_t>(-4.0f) == 0);
  ASSERT_TRUE(Clamp<uint16_t>(1e+20f) == 0xffff);
  ASSERT_TRUE(Clamp<uint16_t>(-1e+20f) == 0);
  ASSERT_TRUE(Clamp<uint16_t>(1e+200) == 0xffff);
  ASSERT_TRUE(Clamp<uint16_t>(-1e+200) == 0);

  ASSERT_TRUE(Clamp<int16_t>(-4) == -4);
  ASSERT_TRUE(Clamp<int16_t>(-4.2) == -4);
  ASSERT_TRUE(Clamp<int16_t>(4.2) == 4);
  ASSERT_TRUE(Clamp<int16_t>(0x7fff) == 0x7fff);
  ASSERT_TRUE(Clamp<int16_t>(0x8000) == 0x7fff);
  ASSERT_TRUE(Clamp<int16_t>(0x10000) == 0x7fff);
  ASSERT_TRUE(Clamp<int16_t>(-0x8000) == -0x8000);
  ASSERT_TRUE(Clamp<int16_t>(-0x10000) == -0x8000);
  ASSERT_TRUE(Clamp<int16_t>(1e+20f) == 0x7fff);
  ASSERT_TRUE(Clamp<int16_t>(-1e+20f) == -0x8000);
  ASSERT_TRUE(Clamp<int16_t>(1e+200) == 0x7fff);
  ASSERT_TRUE(Clamp<int16_t>(-1e+200) == -0x8000);

  ASSERT_TRUE(Clamp<uint32_t>(0) == 0);
  ASSERT_TRUE(Clamp<uint32_t>(0xffffffffLL) == 0xffffffffLL);
  ASSERT_TRUE(Clamp<uint32_t>(100) == 100);
  ASSERT_TRUE(Clamp<uint32_t>(100.3) == 100);
  ASSERT_TRUE(Clamp<uint32_t>(0x100000000LL) == 0xffffffffLL);
  ASSERT_TRUE(Clamp<uint32_t>(-4) == 0);
  ASSERT_TRUE(Clamp<uint32_t>(-4.0f) == 0);
  ASSERT_TRUE(Clamp<uint32_t>(1e+20f) == 0xffffffffu);
  ASSERT_TRUE(Clamp<uint32_t>(-1.0e+20f) == 0);
  ASSERT_TRUE(Clamp<uint32_t>(1e+200) == 0xffffffffu);
  ASSERT_TRUE(Clamp<uint32_t>(-1.0e+200) == 0);

  ASSERT_TRUE(Clamp<int32_t>(-4) == -4);
  ASSERT_TRUE(Clamp<int32_t>(-4LL) == -4);
  ASSERT_TRUE(Clamp<int32_t>(-4.2) == -4);
  ASSERT_TRUE(Clamp<int32_t>(4.2) == 4);
  ASSERT_TRUE(Clamp<int32_t>(0x7fffffff) == 0x7fffffff);
  ASSERT_TRUE(Clamp<int32_t>(0x80000000L) == 0x7fffffff);
  ASSERT_TRUE(Clamp<int32_t>(0x100000000L) == 0x7fffffff);
  ASSERT_TRUE(Clamp<int32_t>(-0x80000000LL) == -0x7fffffff - 1);
  ASSERT_TRUE(Clamp<int32_t>(-0x100000000LL) == -0x7fffffff - 1);
  ASSERT_TRUE(Clamp<int32_t>(1.0e+20f) == 0x7fffffff);
  ASSERT_TRUE(Clamp<int32_t>(-1.0e+20f) == -0x80000000L);
  ASSERT_TRUE(Clamp<int32_t>(1.0e+200) == 0x7fffffff);
  ASSERT_TRUE(Clamp<int32_t>(-1.0e+200) == -0x80000000L);

  ASSERT_TRUE(Clamp<int64_t>(1.0e+200) == 0x7fffffffffffffffLL);
  ASSERT_TRUE(Clamp<int64_t>(-1.0e+200) == -0x7fffffffffffffffLL - 1);
  ASSERT_TRUE(Clamp<uint64_t>(1.0e+200) == 0xffffffffffffffffULL);
  ASSERT_TRUE(Clamp<uint64_t>(-1.0e+200) == 0);
}

TEST(ConvertSat, float2int) {
  FOR_RANGE(int32_t, exp, -10, 100) {
    FOR_RANGE(float, sig, -256, 257) {
      float f = ldexpf(sig, exp);
      float integral;
      float fract = modff(f, &integral);
      if (fract == 0.5f || fract == -0.5f) continue;
      double rounded = roundf(f);
      int64_t clamped = clamp<double>(rounded, -128, 127);
      ASSERT_EQ(ConvertSat<int8_t>(f), clamped) << " with f = " << f;
      clamped = clamp<double>(rounded, 0, 255);
      ASSERT_EQ(ConvertSat<uint8_t>(f), clamped) << " with f = " << f;
      clamped = clamp<double>(rounded, -0x8000, 0x7fff);
      ASSERT_EQ(ConvertSat<int16_t>(f), clamped) << " with f = " << f;
      clamped = clamp<double>(rounded, 0, 0xffff);
      ASSERT_EQ(ConvertSat<uint16_t>(f), clamped) << " with f = " << f;
      clamped = clamp<double>(rounded, int32_t(~0x7fffffff), 0x7fffffff);
      ASSERT_EQ(ConvertSat<int32_t>(f), clamped) << " with f = " << f;
      clamped = clamp<double>(rounded, 0, 0xffffffffu);
      ASSERT_EQ(ConvertSat<uint32_t>(f), clamped) << " with f = " << f;
    }
  }
}

TEST(ConvertNorm, int2int) {
  EXPECT_EQ((ConvertNorm<uint8_t, uint8_t>(0)), 0);
  EXPECT_EQ((ConvertNorm<uint8_t, int8_t>(127)), 255);
}

TEST(ConvertNorm, float2int) {
  EXPECT_EQ(ConvertNorm<uint8_t>(0.0f), 0);
  EXPECT_EQ(ConvertNorm<uint8_t>(0.499f), 127);
  EXPECT_EQ(ConvertNorm<uint8_t>(1.0f), 255);
  EXPECT_EQ(ConvertNorm<int8_t>(1.0f), 127);
  EXPECT_EQ(ConvertNorm<int8_t>(0.499f), 63);
  EXPECT_EQ(ConvertNorm<int8_t>(-1.0f), -127);

  EXPECT_EQ(ConvertNorm<uint16_t>(0.0f), 0);
  EXPECT_EQ(ConvertNorm<uint16_t>(1.0f), 0xffff);
  EXPECT_EQ(ConvertNorm<int16_t>(1.0f), 0x7fff);
  EXPECT_EQ(ConvertNorm<int16_t>(-1.0f), -0x7fff);
}

TEST(ConvertSatNorm, float2int) {
  EXPECT_EQ(ConvertSatNorm<uint8_t>(2.0f), 255);
  EXPECT_EQ(ConvertSatNorm<uint8_t>(0.499f), 127);
  EXPECT_EQ(ConvertSatNorm<uint8_t>(-2.0f), 0);
  EXPECT_EQ(ConvertSatNorm<int8_t>(2.0f), 127);
  EXPECT_EQ(ConvertSatNorm<int8_t>(0.499f), 63);
  EXPECT_EQ(ConvertSatNorm<int8_t>(-2.0f), -128);
  EXPECT_EQ(ConvertSatNorm<uint8_t>(0.4f / 255), 0);
  EXPECT_EQ(ConvertSatNorm<uint8_t>(0.6f / 255), 1);

  EXPECT_EQ(ConvertSatNorm<int16_t>(2.0f), 0x7fff);
  EXPECT_EQ(ConvertSatNorm<int16_t>(-2.0f), -0x8000);
}

TEST(ConvertNorm, int2float) {
  EXPECT_EQ((ConvertNorm<float, uint8_t>(255)), 1.0f);
  EXPECT_NEAR((ConvertNorm<float, uint8_t>(127)), 1.0f * 127 / 255, 1e-7f);
  EXPECT_EQ((ConvertNorm<float, int8_t>(127)), 1.0f);
  EXPECT_NEAR((ConvertNorm<float, int8_t>(64)), 1.0f * 64 / 127, 1e-7f);
}

TEST(Clamp1, int64_2_float16) {
  int64_t big_num = 0x0FFFFFFFFFFFFFFF;
  EXPECT_EQ(static_cast<float>(Clamp<float16>(big_num)), Clamp<float16>(Clamp<float>(big_num)));
  EXPECT_EQ(65504.0f, Clamp<float16>(big_num));
  EXPECT_EQ(-65504.0f, Clamp<float16>(-big_num));
}

TEST(Clamp2, float16_2_int64) {
  float16 fp16 = static_cast<float16>(65504.0f);
  EXPECT_EQ(65504, Clamp<int64_t>(fp16));
  EXPECT_EQ(-65504, Clamp<int64_t>(-fp16));
}

}  // namespace oneflow