#ifndef ONEFLOW_CORE_RECORD_OFRECORD_BYTES_LIST_ENCODER_H_
#define ONEFLOW_CORE_RECORD_OFRECORD_BYTES_LIST_ENCODER_H_

#include "oneflow/core/record/ofrecord_encoder.h"

namespace oneflow {

template<typename T>
class OFRecordEncoderImpl<EncodeCase::kBytesList, T> final : public OFRecordEncoderIf {
 private:
  void EncodeOneCol(DeviceCtx*, const Blob* in_blob, int64_t in_offset, Feature&,
                    const std::string& field_name, int64_t one_col_elem_num) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RECORD_OFRECORD_BYTES_LIST_ENCODER_H_
