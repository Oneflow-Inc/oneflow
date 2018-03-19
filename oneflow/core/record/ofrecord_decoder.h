#ifndef ONEFLOW_CORE_RECORD_OFRECORD_DECODER_H_
#define ONEFLOW_CORE_RECORD_OFRECORD_DECODER_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

class OFRecordDecoderIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OFRecordDecoderIf);
  virtual ~OFRecordDecoderIf() = default;

  virtual int32_t DecodeOneCol(DeviceCtx*, RecordBlob<OFRecord>*,
                               const std::string& name, int32_t cur_col_id,
                               Blob* out_blob) const = 0;

 protected:
  OFRecordDecoderIf() = default;

 private:
};

template<EncodeType encode_type, typename T>
class OFRecordDecoder : public OFRecordDecoderIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OFRecordDecoder);
  virtual ~OFRecordDecoder() = default;

  int32_t DecodeOneCol(DeviceCtx*, RecordBlob<OFRecord>*,
                       const std::string& name, int32_t cur_col_id,
                       Blob* out_blob) const override;

 protected:
  OFRecordDecoder() = default;
  virtual int32_t GetColNumOfFeature(const Feature&,
                                     int64_t elem_num_of_one_col) = 0;
  virtual void ReadOneRow(DeviceCtx*, const Feature&, int32_t col_id,
                          Blob* out_blob) = 0;

 private:
  // return: max_col_num
  int32_t ReadColNum(DeviceCtx*, RecordBlob<OFRecord>*, const std::string& name,
                     Blob* out_blob);
  void ReadDataId(DeviceCtx*, RecordBlob<OFRecord>*, Blob* out_blob);
  void ReadDataContent(DeviceCtx*, RecordBlob<OFRecord>*,
                       const std::string& name, int32_t col_id, Blob* out_blob);
};

template<EncodeType encode_type, typename T>
class OFRecordDecoderImpl;

OFRecordDecoderIf* GetOFRecordDecoder(EncodeType, DataType);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RECORD_OFRECORD_DECODER_H_
