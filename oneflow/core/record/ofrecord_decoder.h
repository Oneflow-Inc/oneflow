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
#ifndef ONEFLOW_CORE_RECORD_OFRECORD_DECODER_H_
#define ONEFLOW_CORE_RECORD_OFRECORD_DECODER_H_

#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/operator/op_conf_util.h"

namespace oneflow {

using EncodeCase = EncodeConf::EncodeCase;

class OFRecordDecoderIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OFRecordDecoderIf);
  virtual ~OFRecordDecoderIf() = default;

  virtual int32_t DecodeOneCol(DeviceCtx*, Blob* in_blob, const BlobConf&, int32_t cur_col_id,
                               Blob* out_blob,
                               std::function<int32_t(void)> NextRandomInt) const = 0;
  virtual bool HasDim1ValidNumField(const EncodeConf& encode_conf) const = 0;
  virtual bool HasDim2ValidNumField(const EncodeConf& encode_conf) const = 0;

 protected:
  OFRecordDecoderIf() = default;

 private:
};

template<EncodeCase encode_case, typename T>
class OFRecordDecoder : public OFRecordDecoderIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OFRecordDecoder);
  virtual ~OFRecordDecoder() = default;

  int32_t DecodeOneCol(DeviceCtx*, Blob* in_blob, const BlobConf&, int32_t cur_col_id,
                       Blob* out_blob, std::function<int32_t(void)> NextRandomInt) const override;

 protected:
  OFRecordDecoder() = default;
  virtual int32_t GetColNumOfFeature(const Feature&, int64_t one_col_elem_num) const = 0;
  virtual void ReadOneCol(DeviceCtx*, const Feature&, const BlobConf&, int32_t col_id, T* out_dptr,
                          int64_t one_col_elem_num,
                          std::function<int32_t(void)> NextRandomInt) const = 0;
  virtual void SetDim1ValidNum(const Feature& feature, Blob* out_blob, int64_t dim0_idx) const {
    UNIMPLEMENTED();
  }
  virtual void SetDim2ValidNum(const Feature& feature, Blob* out_blob, int64_t dim0_idx) const {
    UNIMPLEMENTED();
  }

 private:
  // return: max_col_num
  int32_t ReadColNum(DeviceCtx*, Blob*, const std::string& name, Blob* out_blob) const;
  void ReadDataId(DeviceCtx*, Blob* in_blob, Blob* out_blob) const;
  void ReadDataContent(DeviceCtx*, Blob* in_blob, const BlobConf&, int32_t col_id, Blob* out_blob,
                       std::function<int32_t(void)> NextRandomInt) const;
  void ReadPartDataContent(DeviceCtx*, Blob* in_blob, const BlobConf&, int32_t col_id,
                           Blob* out_blob, int32_t part_id, int32_t part_num,
                           int64_t one_col_elem_num, int32_t random_seed) const;
};

template<typename T>
void DoPreprocess(const PreprocessConf& conf, T* dptr, const Shape& shape);

template<EncodeCase encode_case, typename T>
class OFRecordDecoderImpl;

OFRecordDecoderIf* GetOFRecordDecoder(EncodeCase, DataType);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RECORD_OFRECORD_DECODER_H_
