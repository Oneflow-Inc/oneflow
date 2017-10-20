#include "oneflow/core/persistence/record_in_stream.h"
#include "oneflow/core/persistence/data_set_util.h"
namespace oneflow {

void RecordInStream::InitHeader() {
  int ret = in_stream_->Read(reinterpret_cast<char*>(header_.get()),
                             sizeof(DataSetHeader));
  CHECK(!ret);
  CHECK(header()->magic_code_ == 0xfeed);
  CHECK(!DataSetUtil::ValidateHeader(*header()));
}

int32_t RecordInStream::ReadRecord(
    std::unique_ptr<Record, decltype(&free)>* buffer) {
  auto buffer_meta = FlexibleMalloc<Record>(0);
  int ret = ReadMeta(reinterpret_cast<char*>(buffer_meta.get()),
                     FlexibleSizeOf<Record>(0));
  if (ret < 0) { return ret; }
  CHECK(!DataSetUtil::ValidateRecord(*buffer_meta));
  *buffer = FlexibleMalloc<Record>(buffer_meta->len_);
  memcpy(reinterpret_cast<char*>((*buffer).get()),
         reinterpret_cast<char*>(buffer_meta.get()), FlexibleSizeOf<Record>(0));
  ret = in_stream_->Read((*buffer)->data_, (*buffer)->len_);
  CHECK(!ret);
  CHECK(!DataSetUtil::ValidateRecord(**buffer));
  return ret;
}

}  // namespace oneflow
