#ifndef ONEFLOW_CORE_RECORD_OFRECORD_RAW_ENCODER_H_
#define ONEFLOW_CORE_RECORD_OFRECORD_RAW_ENCODER_H_

#include "oneflow/core/record/ofrecord_encoder.h"

namespace oneflow {

template<typename T>
class OFRecordEncoderImpl<EncodeCase::kRaw, T> final
    : public OFRecordEncoder<EncodeCase::kRaw, T> {
 public:
 private:
  void EncodeOneCol(DeviceCtx*, const T* in_dptr, Feature&,
                    const std::string& field_name,
                    int64_t one_col_elem_num) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RECORD_OFRECORD_RAW_ENCODER_H_
