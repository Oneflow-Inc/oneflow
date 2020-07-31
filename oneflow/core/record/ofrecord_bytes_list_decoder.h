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
#ifndef ONEFLOW_CORE_RECORD_OFRECORD_BYTES_LIST_DECODER_H_
#define ONEFLOW_CORE_RECORD_OFRECORD_BYTES_LIST_DECODER_H_

#include "oneflow/core/record/ofrecord_decoder.h"

namespace oneflow {

template<typename T>
class OFRecordDecoderImpl<EncodeCase::kBytesList, T> final
    : public OFRecordDecoder<EncodeCase::kBytesList, T> {
 public:
  bool HasDim1ValidNumField(const EncodeConf& encode_conf) const override { return true; }
  bool HasDim2ValidNumField(const EncodeConf& encode_conf) const override { return true; }

 private:
  int32_t GetColNumOfFeature(const Feature&, int64_t) const override { return 1; }
  void ReadOneCol(DeviceCtx*, const Feature&, const BlobConf& blob_conf, int32_t col_id,
                  T* out_dptr, int64_t one_col_elem_num,
                  std::function<int32_t(void)> NextRandomInt) const override;
  void SetDim1ValidNum(const Feature& feature, Blob* out_blob, int64_t dim0_idx) const override;
  void SetDim2ValidNum(const Feature& feature, Blob* out_blob, int64_t dim0_idx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RECORD_OFRECORD_BYTES_LIST_DECODER_H_
