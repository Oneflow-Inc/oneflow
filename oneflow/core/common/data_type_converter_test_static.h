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
#ifndef ONEFLOW_CORE_COMMON_DATA_TYPE_CONVERTER_TEST_STATIC_H_
#define ONEFLOW_CORE_COMMON_DATA_TYPE_CONVERTER_TEST_STATIC_H_

#include "oneflow/core/common/data_type_converter.h"

namespace oneflow {

namespace {
// fp to int
static_assert(NeedsClamp<float, int8_t>::value, "Float range exceeds all ints up to 64b");
static_assert(NeedsClamp<float, uint8_t>::value, "Float range exceeds all ints up to 64b");
static_assert(NeedsClamp<float, int16_t>::value, "Float range exceeds all ints up to 64b");
static_assert(NeedsClamp<float, uint16_t>::value, "Float range exceeds all ints up to 64b");
static_assert(NeedsClamp<float, int32_t>::value, "Float range exceeds all ints up to 64b");
static_assert(NeedsClamp<float, uint32_t>::value, "Float range exceeds all ints up to 64b");
static_assert(NeedsClamp<float, int64_t>::value, "Float range exceeds all ints up to 64b");
static_assert(NeedsClamp<float, uint64_t>::value, "Float range exceeds all ints up to 64b");

// same size, different signedness
static_assert(NeedsClamp<int8_t, uint8_t>::value, "Signed <-> unsigned requires clamp");
static_assert(NeedsClamp<uint8_t, int8_t>::value, "Signed <-> unsigned requires clamp");
static_assert(NeedsClamp<int16_t, uint16_t>::value, "Signed <-> unsigned requires clamp");
static_assert(NeedsClamp<uint16_t, int16_t>::value, "Signed <-> unsigned requires clamp");
static_assert(NeedsClamp<int32_t, uint32_t>::value, "Signed <-> unsigned requires clamp");
static_assert(NeedsClamp<uint32_t, int32_t>::value, "Signed <-> unsigned requires clamp");
static_assert(NeedsClamp<int64_t, uint64_t>::value, "Signed <-> unsigned requires clamp");
static_assert(NeedsClamp<uint64_t, int64_t>::value, "Signed <-> unsigned requires clamp");

// larger, but unsigned
static_assert(NeedsClamp<int8_t, uint16_t>::value, "Need to clamp negatives to 0");
static_assert(NeedsClamp<int8_t, uint32_t>::value, "Need to clamp negatives to 0");
static_assert(NeedsClamp<int8_t, uint64_t>::value, "Need to clamp negatives to 0");
static_assert(NeedsClamp<int16_t, uint32_t>::value, "Need to clamp negatives to 0");
static_assert(NeedsClamp<int16_t, uint64_t>::value, "Need to clamp negatives to 0");
static_assert(NeedsClamp<int32_t, uint64_t>::value, "Need to clamp negatives to 0");

static_assert(!NeedsClamp<int8_t, int8_t>::value, "Clamping not required");
static_assert(!NeedsClamp<int8_t, int16_t>::value, "Clamping not required");
static_assert(!NeedsClamp<uint8_t, int16_t>::value, "Clamping not required");
static_assert(!NeedsClamp<uint8_t, uint16_t>::value, "Clamping not required");
static_assert(!NeedsClamp<float, float>::value, "Clamping not required");
static_assert(!NeedsClamp<float, double>::value, "Clamping not required");

static_assert(Clamp<uint8_t>(0) == 0, "Unexpected clamp result");
static_assert(Clamp<uint8_t>(255) == 255, "Unexpected clamp result");
static_assert(Clamp<uint8_t>(100) == 100, "Unexpected clamp result");
static_assert(Clamp<uint8_t>(100.3) == 100, "Unexpected clamp result");
static_assert(Clamp<uint8_t>(256) == 255, "Unexpected clamp result");
static_assert(Clamp<uint8_t>(-4) == 0, "Unexpected clamp result");
static_assert(Clamp<uint8_t>(-4.0f) == 0, "Unexpected clamp result");
static_assert(Clamp<uint8_t>(1e+20f) == 255, "Unexpected clamp result");
static_assert(Clamp<uint8_t>(-1e+20f) == 0, "Unexpected clamp result");
static_assert(Clamp<uint8_t>(1e+200) == 255, "Unexpected clamp result");
static_assert(Clamp<uint8_t>(-1e+200) == 0, "Unexpected clamp result");

static_assert(Clamp<int8_t>(-4) == -4, "Unexpected clamp result");
static_assert(Clamp<int8_t>(-4.2) == -4, "Unexpected clamp result");
static_assert(Clamp<int8_t>(4.2) == 4, "Unexpected clamp result");
static_assert(Clamp<int8_t>(127) == 127, "Unexpected clamp result");
static_assert(Clamp<int8_t>(128) == 127, "Unexpected clamp result");
static_assert(Clamp<int8_t>(256) == 127, "Unexpected clamp result");
static_assert(Clamp<int8_t>(-128) == -128, "Unexpected clamp result");
static_assert(Clamp<int8_t>(-256) == -128, "Unexpected clamp result");
static_assert(Clamp<int8_t>(1e+20f) == 127, "Unexpected clamp result");
static_assert(Clamp<int8_t>(-1e+20f) == -128, "Unexpected clamp result");
static_assert(Clamp<int8_t>(1e+200) == 127, "Unexpected clamp result");
static_assert(Clamp<int8_t>(-1e+200) == -128, "Unexpected clamp result");

static_assert(Clamp<uint16_t>(0) == 0, "Unexpected clamp result");
static_assert(Clamp<uint16_t>(0xffff) == 0xffff, "Unexpected clamp result");
static_assert(Clamp<uint16_t>(100) == 100, "Unexpected clamp result");
static_assert(Clamp<uint16_t>(100.3) == 100, "Unexpected clamp result");
static_assert(Clamp<uint16_t>(0x10000) == 0xffff, "Unexpected clamp result");
static_assert(Clamp<uint16_t>(-4) == 0, "Unexpected clamp result");
static_assert(Clamp<uint16_t>(-4.0f) == 0, "Unexpected clamp result");
static_assert(Clamp<uint16_t>(1e+20f) == 0xffff, "Unexpected clamp result");
static_assert(Clamp<uint16_t>(-1e+20f) == 0, "Unexpected clamp result");
static_assert(Clamp<uint16_t>(1e+200) == 0xffff, "Unexpected clamp result");
static_assert(Clamp<uint16_t>(-1e+200) == 0, "Unexpected clamp result");

static_assert(Clamp<int16_t>(-4) == -4, "Unexpected clamp result");
static_assert(Clamp<int16_t>(-4.2) == -4, "Unexpected clamp result");
static_assert(Clamp<int16_t>(4.2) == 4, "Unexpected clamp result");
static_assert(Clamp<int16_t>(0x7fff) == 0x7fff, "Unexpected clamp result");
static_assert(Clamp<int16_t>(0x8000) == 0x7fff, "Unexpected clamp result");
static_assert(Clamp<int16_t>(0x10000) == 0x7fff, "Unexpected clamp result");
static_assert(Clamp<int16_t>(-0x8000) == -0x8000, "Unexpected clamp result");
static_assert(Clamp<int16_t>(-0x10000) == -0x8000, "Unexpected clamp result");
static_assert(Clamp<int16_t>(1e+20f) == 0x7fff, "Unexpected clamp result");
static_assert(Clamp<int16_t>(-1e+20f) == -0x8000, "Unexpected clamp result");
static_assert(Clamp<int16_t>(1e+200) == 0x7fff, "Unexpected clamp result");
static_assert(Clamp<int16_t>(-1e+200) == -0x8000, "Unexpected clamp result");

static_assert(Clamp<uint32_t>(0) == 0, "Unexpected clamp result");
static_assert(Clamp<uint32_t>(0xffffffffLL) == 0xffffffffLL, "Unexpected clamp result");
static_assert(Clamp<uint32_t>(100) == 100, "Unexpected clamp result");
static_assert(Clamp<uint32_t>(100.3) == 100, "Unexpected clamp result");
static_assert(Clamp<uint32_t>(0x100000000LL) == 0xffffffffLL, "Unexpected clamp result");
static_assert(Clamp<uint32_t>(-4) == 0, "Unexpected clamp result");
static_assert(Clamp<uint32_t>(-4.0f) == 0, "Unexpected clamp result");
static_assert(Clamp<uint32_t>(1e+20f) == 0xffffffffu, "Unexpected clamp result");
static_assert(Clamp<uint32_t>(-1.0e+20f) == 0, "Unexpected clamp result");
static_assert(Clamp<uint32_t>(1e+200) == 0xffffffffu, "Unexpected clamp result");
static_assert(Clamp<uint32_t>(-1.0e+200) == 0, "Unexpected clamp result");

static_assert(Clamp<int32_t>(-4) == -4, "Unexpected clamp result");
static_assert(Clamp<int32_t>(-4LL) == -4, "Unexpected clamp result");
static_assert(Clamp<int32_t>(-4.2) == -4, "Unexpected clamp result");
static_assert(Clamp<int32_t>(4.2) == 4, "Unexpected clamp result");
static_assert(Clamp<int32_t>(0x7fffffff) == 0x7fffffff, "Unexpected clamp result");
static_assert(Clamp<int32_t>(0x80000000L) == 0x7fffffff, "Unexpected clamp result");
static_assert(Clamp<int32_t>(0x100000000L) == 0x7fffffff, "Unexpected clamp result");
static_assert(Clamp<int32_t>(-0x80000000LL) == -0x7fffffff - 1, "Unexpected clamp result");
static_assert(Clamp<int32_t>(-0x100000000LL) == -0x7fffffff - 1, "Unexpected clamp result");
static_assert(Clamp<int32_t>(1.0e+20f) == 0x7fffffff, "Unexpected clamp result");
static_assert(Clamp<int32_t>(-1.0e+20f) == -0x80000000L, "Unexpected clamp result");
static_assert(Clamp<int32_t>(1.0e+200) == 0x7fffffff, "Unexpected clamp result");
static_assert(Clamp<int32_t>(-1.0e+200) == -0x80000000L, "Unexpected clamp result");

static_assert(Clamp<int64_t>(1.0e+200) == 0x7fffffffffffffffLL, "Unexpected clamp result");
static_assert(Clamp<int64_t>(-1.0e+200) == -0x7fffffffffffffffLL - 1, "Unexpected clamp result");
static_assert(Clamp<uint64_t>(1.0e+200) == 0xffffffffffffffffULL, "Unexpected clamp result");
static_assert(Clamp<uint64_t>(-1.0e+200) == 0, "Unexpected clamp result");

}  // namespace
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_DATA_TYPE_CONVERTER_TEST_STATIC_H_