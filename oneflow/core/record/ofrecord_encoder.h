#ifndef ONEFLOW_CORE_RECORD_OFRECORD_ENCODER_H_
#define ONEFLOW_CORE_RECORD_OFRECORD_ENCODER_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

using EncodeCase = PrintOpConf::EncodeCase;

class OFRecordEncoderIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OFRecordEncoderIf);
  virtual ~OFRecordEncoderIf() = default;

  virtual void EncodeOneFieldToOneRecord(DeviceCtx*, int64_t record_id,
                                         const Blob*,
                                         const std::string& field_name,
                                         OFRecord&) const = 0;
  static void EncodeDataIdToOneRecord(DeviceCtx* ctx, const char* data_id_str,
                                      OFRecord& record) {
    record.mutable_feature()->at("data_id").mutable_bytes_list()->add_value(
        data_id_str);
  }

 protected:
  OFRecordEncoderIf() = default;

 private:
};

template<EncodeCase encode_case, typename T>
class OFRecordEncoder : public OFRecordEncoderIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OFRecordEncoder);
  virtual ~OFRecordEncoder() = default;

  void EncodeOneFieldToOneRecord(DeviceCtx*, int64_t record_id, const Blob*,
                                 const std::string& field_name,
                                 OFRecord&) const override;

 protected:
  OFRecordEncoder() = default;
  virtual void EncodeOneCol(DeviceCtx*, const T* in_dptr, Feature&,
                            const std::string& field_name,
                            int64_t one_col_elem_num) const = 0;
};

template<EncodeCase encode_case, typename T>
class OFRecordEncoderImpl;

#define ENCODE_CASE_SEQ                  \
  OF_PP_MAKE_TUPLE_SEQ(EncodeCase::kRaw) \
  OF_PP_MAKE_TUPLE_SEQ(EncodeCase::kJpeg)

OFRecordEncoderIf* GetOFRecordEncoder(EncodeCase, DataType);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RECORD_OFRECORD_ENCODER_H_
