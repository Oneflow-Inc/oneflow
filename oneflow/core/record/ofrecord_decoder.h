#ifndef ONEFLOW_CORE_RECORD_OFRECORD_DECODER_H_
#define ONEFLOW_CORE_RECORD_OFRECORD_DECODER_H_

#include "oneflow/core/register/blob.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

class OFRecordDecoderIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OFRecordDecoderIf);
  virtual ~OFRecordDecoderIf() = default;

  virtual int32_t Decode(RecordBlob<OFRecord>*, const std::string& name,
                         int32_t col_id, Blob* out_blob, DeviceCtx*) = 0;

 protected:
  OFRecordDecoderIf() = default;
};

template<EncodeType, DataType>
class OFRecordDecoder : public OFRecordDecoderIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OFRecordDecoder);
  ~OFRecordDecoder() = default;

  int32_t Decode(RecordBlob<OFRecord>*, const std::string& name, int32_t col_id,
                 Blob* out_blob, DeviceCtx*) override;

 protected:
  OFRecordDecoder() = default;
  int32_t ReadColNumToOutBlob(RecordBlob<OFRecord>*, const std::string& name,
                              Blob* out_blob);
  void ReadDataIdToOutBlob(RecordBlob<OFRecord>*, Blob* out_blob,
                           DeviceCtx* ctx);
  void ReadDataContentToOutBlob(RecordBlob<OFRecord>*, const std::string& name,
                                int32_t col_id, Blob* out_blob, DeviceCtx*);

 private:
  virtual int32_t GetColNumOfFeature(const Feature&, int64_t item_size) = 0;
  virtual void ReadDataContentForOneItem(const Feature&, int32_t col_id,
                                         Blob* out_blob, DeviceCtx*) = 0;
};

DataType DataTypeOf(const Feature& feature);
int64_t SizeOf(const Feature& feature);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RECORD_OFRECORD_DECODER_H_
