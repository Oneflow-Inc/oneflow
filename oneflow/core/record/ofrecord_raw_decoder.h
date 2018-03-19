#ifndef ONEFLOW_CORE_RECORD_OFRECORD_RAW_DECODER_H_
#define ONEFLOW_CORE_RECORD_OFRECORD_RAW_DECODER_H_

#include "oneflow/core/record/ofrecord_decoder.h"

namespace oneflow {

template<typename T>
class OFRecordDecoderImpl<EncodeType::kRaw, T> final
    : public OFRecordDecoder<EncodeType::kRaw, T> {
 public:
 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RECORD_OFRECORD_RAW_DECODER_H_
