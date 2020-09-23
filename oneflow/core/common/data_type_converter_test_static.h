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

}  // namespace
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_DATA_TYPE_CONVERTER_TEST_STATIC_H_
