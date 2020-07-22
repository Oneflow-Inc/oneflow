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
#ifndef ONEFLOW_CORE_RECORD_ENCODE_CASE_UTIL_H_
#define ONEFLOW_CORE_RECORD_ENCODE_CASE_UTIL_H_

#include "oneflow/core/common/data_type.h"

namespace oneflow {

//  encode case
#define ENCODE_CASE_DATA_TYPE_SEQ_PRODUCT                                            \
  OF_PP_SEQ_PRODUCT((EncodeCase::kJpeg), ARITHMETIC_DATA_TYPE_SEQ)                   \
  OF_PP_SEQ_PRODUCT((EncodeCase::kRaw), ARITHMETIC_DATA_TYPE_SEQ CHAR_DATA_TYPE_SEQ) \
  OF_PP_SEQ_PRODUCT((EncodeCase::kBytesList), ((char, DataType::kChar))((int8_t, DataType::kInt8)))

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RECORD_ENCODE_CASE_UTIL_H_
