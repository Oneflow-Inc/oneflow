#ifndef ONEFLOW_CORE_RECORD_OFRECORD_RAW_DECODER_H_
#define ONEFLOW_CORE_RECORD_OFRECORD_RAW_DECODER_H_

#include "oneflow/core/record/ofrecord_decoder.h"

namespace oneflow {

template<typename T>
class OFRecordDecoderImpl<EncodeCase::kRaw, T> final
    : public OFRecordDecoder<EncodeCase::kRaw, T> {
 public:
 private:
  int32_t GetColNumOfFeature(const Feature&,
                             int64_t one_col_elem_num) const override;
  void ReadOneCol(DeviceCtx*, const Feature&, const BlobConf& blob_conf,
                  int32_t col_id, T* out_dptr, int64_t one_col_elem_num,
                  std::function<int32_t(void)> NextRandomInt) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RECORD_OFRECORD_RAW_DECODER_H_
