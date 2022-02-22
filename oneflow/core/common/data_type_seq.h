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
#ifndef ONEFLOW_CORE_COMMON_DATA_TYPE_SEQ_H_
#define ONEFLOW_CORE_COMMON_DATA_TYPE_SEQ_H_

#include "oneflow/core/common/preprocessor.h"

// SEQ

#define BOOL_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(bool, DataType::kBool)

#define FLOATING_DATA_TYPE_SEQ                  \
  OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat) \
  OF_PP_MAKE_TUPLE_SEQ(double, DataType::kDouble)

#define SIGNED_INT_DATA_TYPE_SEQ                  \
  OF_PP_MAKE_TUPLE_SEQ(int8_t, DataType::kInt8)   \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)

#define UNSIGNED_INT_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(uint8_t, DataType::kUInt8)

#define INT_DATA_TYPE_SEQ SIGNED_INT_DATA_TYPE_SEQ

#define CHAR_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(char, DataType::kChar)

#define ARITHMETIC_DATA_TYPE_SEQ \
  FLOATING_DATA_TYPE_SEQ         \
  INT_DATA_TYPE_SEQ

#define POD_DATA_TYPE_SEQ \
  ARITHMETIC_DATA_TYPE_SEQ CHAR_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ
#define POD_AND_HALF_DATA_TYPE_SEQ POD_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ
#define PB_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(OFRecord, DataType::kOFRecord)
#define ALL_DATA_TYPE_SEQ POD_DATA_TYPE_SEQ PB_DATA_TYPE_SEQ

#define INDEX_DATA_TYPE_SEQ                       \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)

#define FLOAT16_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(float16, DataType::kFloat16)

#if defined(WITH_CUDA)
#define HALF_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(half, DataType::kFloat16)
#endif

#define IMAGE_DATA_TYPE_SEQ                       \
  OF_PP_MAKE_TUPLE_SEQ(uint8_t, DataType::kUInt8) \
  OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat)

#endif  // ONEFLOW_CORE_COMMON_DATA_TYPE_SEQ_H_
