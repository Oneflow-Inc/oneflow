#include "oneflow/core/persistence/data_set_in_stream.h"
#include "oneflow/core/persistence/data_set_util.h"
namespace oneflow {

void DataSetInStream::InitHeader() {
  CHECK(file_size());
  CHECK(!cur_file_pos());
  Read(reinterpret_cast<char*>(header_.get()), sizeof(DataSetHeader));
  CHECK(header()->magic_code == 0xfeed);
  CHECK(!DataSetUtil::ValidateHeader(*header()));
}

int32_t DataSetInStream::ReadBuffer(
    std::unique_ptr<Buffer, decltype(&free)>* buffer) {
  auto buffer_meta = FlexibleMalloc<Buffer>(0);
  int ret = Read(reinterpret_cast<char*>(buffer_meta.get()),
                 FlexibleSizeOf<Buffer>(0));
  if (ret < 0) { return ret; }
  CHECK(!DataSetUtil::ValidateBuffer(*buffer_meta));
  *buffer = FlexibleMalloc<Buffer>(buffer_meta->len);
  memcpy(reinterpret_cast<char*>((*buffer).get()),
         reinterpret_cast<char*>(buffer_meta.get()), FlexibleSizeOf<Buffer>(0));
  ret = Read((*buffer)->data, (*buffer)->len);
  CHECK(!DataSetUtil::ValidateBuffer(**buffer));
  return ret;
}

}  // namespace oneflow
