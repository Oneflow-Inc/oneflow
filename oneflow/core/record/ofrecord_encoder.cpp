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
#include "oneflow/core/record/ofrecord_encoder.h"
#include "oneflow/core/record/ofrecord_raw_encoder.h"
#include "oneflow/core/record/ofrecord_jpeg_encoder.h"
#include "oneflow/core/record/ofrecord_bytes_list_encoder.h"
#include "oneflow/core/record/encode_case_util.h"

namespace oneflow {

// TODO(niuchong): use ENCODE_CASE_DATA_TYPE_SEQ_PRODUCT
#define TMP_ENCODE_CASE_DATA_TYPE_SEQ_PRODUCT                                          \
  OF_PP_SEQ_PRODUCT((EncodeCase::kJpeg), ARITHMETIC_DATA_TYPE_SEQ)                     \
  OF_PP_SEQ_PRODUCT((EncodeCase::kRaw),                                                \
                    ARITHMETIC_DATA_TYPE_SEQ CHAR_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ) \
  OF_PP_SEQ_PRODUCT((EncodeCase::kBytesList), ((char, DataType::kChar))((int8_t, DataType::kInt8)))

OFRecordEncoderIf* GetOFRecordEncoder(EncodeCase encode_case, DataType data_type) {
  static const HashMap<std::string, OFRecordEncoderIf*> obj = {
#define MAKE_ENTRY(et, dt) \
  {GetHashKey(et, OF_PP_PAIR_SECOND(dt)), new OFRecordEncoderImpl<et, OF_PP_PAIR_FIRST(dt)>},
      OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, TMP_ENCODE_CASE_DATA_TYPE_SEQ_PRODUCT)};
  return obj.at(GetHashKey(encode_case, data_type));
}

}  // namespace oneflow
