#include "oneflow/core/record/record_io.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

int32_t ReadRecord(PersistentInStream* in_stream, Blob* records) {
  int64_t record_size = -1;
  int64_t record_num = records->shape().elem_cnt();
  for (size_t i = 0; i < record_num; ++i) {
    if (in_stream->Read(reinterpret_cast<char*>(&record_size), sizeof(int64_t)) == 0) {
      std::unique_ptr<char[]> buffer(new char[record_size]);
      CHECK_EQ(in_stream->Read(buffer.get(), record_size), 0);
      if (records->blob_desc().body().data_type() == kOFRecord) {
        (records->mut_dptr<OFRecord>() + i)->ParseFromArray(buffer.get(), record_size);
      } else {
        UNIMPLEMENTED();
      }
    } else {
      return i;
    }
  }
  return record_num;
}

}  // namespace oneflow
