#ifndef ONEFLOW_CORE_RECORD_RAW_OFRECORD_DECODER_H_
#define ONEFLOW_CORE_RECORD_RAW_OFRECORD_DECODER_H_

#include "oneflow/core/record/ofrecord_decoder.h"

namespace oneflow {

template<DataType D>
class RawOFRecordDecoder final : public OFRecordDecoder<EncodeType::kRaw, D> {
 private:
  int32_t GetColNumOfFeature(const Feature&, int64_t item_size) override;
  void ReadDataContentForOneItem(const Feature&, int32_t col_id, Blob* out_blob,
                                 DeviceCtx*) override;

  void ReadWithSameDataType(const Feature&, int32_t col_id, Blob* out_blob,
                            DeviceCtx*);
  template<typename T>
  void ReadFromInDptrToOutDptr(const T* in_dptr, Blob* out_blob,
                               int32_t col_id);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RECORD_RAW_OFRECORD_DECODER_H_
