#include "oneflow/core/persistence/data_set_in_stream.h"
#include "oneflow/core/persistence/data_set_util.h"
namespace oneflow {

void DataSetInStream::InitHeader() {
  CHECK(file_size());
  CHECK(!cur_file_pos());
  Read(reinterpret_cast<char*>(header_.get()), sizeof(DataSetHeader));
  CHECK(header()->magic_code_ == 0xfeed);
  CHECK(!DataSetUtil::ValidateHeader(*header()));
}

int32_t DataSetInStream::ReadRecord(
    std::unique_ptr<Record, decltype(&free)>* buffer) {
  auto buffer_meta = FlexibleMalloc<Record>(0);
  int ret = Read(reinterpret_cast<char*>(buffer_meta.get()),
                 FlexibleSizeOf<Record>(0));
  if (ret < 0) { return ret; }
  CHECK(!DataSetUtil::ValidateRecord(*buffer_meta));
  *buffer = FlexibleMalloc<Record>(buffer_meta->len_);
  memcpy(reinterpret_cast<char*>((*buffer).get()),
         reinterpret_cast<char*>(buffer_meta.get()), FlexibleSizeOf<Record>(0));
  ret = Read((*buffer)->data_, (*buffer)->len_);
  CHECK(!DataSetUtil::ValidateRecord(**buffer));
  return ret;
}

}  // namespace oneflow
