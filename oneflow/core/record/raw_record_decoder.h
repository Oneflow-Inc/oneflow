#ifndef ONEFLOW_CORE_RECORD_RAW_RECORD_DECODER_H_
#define ONEFLOW_CORE_RECORD_RAW_RECORD_DECODER_H_

#include "oneflow/core/record/record_decoder.h"

namespace oneflow {

template<typename T>
class RawRecordDecoder final : public RecordDecoder<T> {
 private:
  int32_t GetColNumOfFeature(const Feature&, int64_t item_size) override;
  void ReadDataContentForOneItem(const Feature&, int32_t cur_col_id,
                                 T* out_dptr, int64_t item_size,
                                 DeviceCtx*) override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RECORD_RAW_RECORD_DECODER_H_
