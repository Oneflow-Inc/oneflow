#ifndef ONEFLOW_CORE_RECORD_RECORD_DECODER_H_
#define ONEFLOW_CORE_RECORD_RECORD_DECODER_H_

#include "oneflow/core/register/blob.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/record/feature_list_util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<typename T>
class RecordDecoder {
 public:
  RecordDecoder() = default;
  ~RecordDecoder() = default;

  int32_t ReadRecordToOutBlob(RecordBlob<OFRecord>*, const std::string& name,
                              int32_t cur_col_id, Blob* out_blob, DeviceCtx*);

 protected:
  virtual int32_t GetColNumOfFeature(Feature&, int64_t item_size) = 0;
  virtual void ReadDataContentForOneItem(T* dptr, Feature&, int64_t item_size,
                                         DeviceCtx*) = 0;

  int32_t ReadColNumToOutBlob(RecordBlob<OFRecord>*, const std::string& name,
                              Blob* out_blob);
  void ReadDataIdToOutBlob(RecordBlob<OFRecord>*, Blob* out_blob,
                           DeviceCtx* ctx);
  void ReadDataContentToOutBlob(RecordBlob<OFRecord>*, const std::string& name,
                                int32_t cur_col_id, Blob* out_blob, DeviceCtx*);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RECORD_RECORD_DECODER_H_
