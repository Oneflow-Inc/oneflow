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
#ifndef ONEFLOW_CORE_RECORD_OFRECORD_ENCODER_H_
#define ONEFLOW_CORE_RECORD_OFRECORD_ENCODER_H_

#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/record/ofrecord_decoder.h"

namespace oneflow {

class OFRecordEncoderIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OFRecordEncoderIf);
  virtual ~OFRecordEncoderIf() = default;

  static void EncodeOneDataId(DeviceCtx* ctx, const char* data_id_str, OFRecord& record) {
    Feature tmp_feature;
    tmp_feature.mutable_bytes_list()->add_value(data_id_str);
    CHECK(record.mutable_feature()->insert({"data_id", tmp_feature}).second);
  }
  virtual void EncodeOneCol(DeviceCtx*, const Blob* in_blob, int64_t in_offset, Feature&,
                            const std::string& field_name, int64_t one_col_elem_num) const = 0;

 protected:
  OFRecordEncoderIf() = default;
};

template<EncodeCase encode_case, typename T>
class OFRecordEncoderImpl;

OFRecordEncoderIf* GetOFRecordEncoder(EncodeCase, DataType);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RECORD_OFRECORD_ENCODER_H_
