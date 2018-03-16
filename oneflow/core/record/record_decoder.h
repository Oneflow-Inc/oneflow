#include "oneflow/core/register/blob.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<typename T>
class RecordDecoder {
 public:
  RecordDecoder() = default;
  ~RecordDecoder() = default;

  int32_t ReadRecordToOutBlob(RecordBlob<OFRecord>*, const std::string& name,
                              int32_t cur_col_id, Blob* out_blob, DeviceCtx*);

 protectd:
  virtual int64_t GetUnitSizeInRecordBlob();
  
  int32_t ReadColNumToOutBlob();
  void ReadDataIdToOutBlob();
  void ReadDataContentToOutBlob();

  RecordBlob<OFRecord>* record_blob;
  std::string& name;
  int32_t cur_col_id;
  Blob* out_blob;
  DeviceCtx* ctx;
};

}  // namespace oneflow
