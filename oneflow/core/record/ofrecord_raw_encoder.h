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
#ifndef ONEFLOW_CORE_RECORD_OFRECORD_RAW_ENCODER_H_
#define ONEFLOW_CORE_RECORD_OFRECORD_RAW_ENCODER_H_

#include "oneflow/core/record/ofrecord_encoder.h"

namespace oneflow {

template<typename T>
class OFRecordEncoderImpl<EncodeCase::kRaw, T> final : public OFRecordEncoderIf {
 public:
  void EncodeBlob(DeviceCtx* ctx, const Blob* in_blob, Feature* feature) const;

 private:
  void EncodeOneCol(DeviceCtx*, const Blob* in_blob, int64_t in_offset, Feature&,
                    const std::string& field_name, int64_t one_col_elem_num) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RECORD_OFRECORD_RAW_ENCODER_H_
