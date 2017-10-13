#include "oneflow/core/kernel/data_set_in_stream.h"
#include "oneflow/core/kernel/data_set_util.h"
namespace oneflow {

void DataSetInStream::InitHeader() {
  CHECK(file_size());
  CHECK(!cur_file_pos());
  Read(reinterpret_cast<char*>(header_.get()), sizeof(*header_));
  CHECK(header()->magic_code == 0xfeed);
  CHECK(file_size()
        == header()->DataBodyOffset()
               + +header()->data_item_count
                     * FlexibleSizeOf<DataItem>(header()->TensorElemCount()));
}

void DataSetInStream::SkipLabelDesc() {
  CHECK(cur_file_pos() == sizeof(DataSetHeader));
  if (header()->label_desc_buf_len) {
    set_cur_file_pos(cur_file_pos() + header()->label_desc_buf_len);
  }
}

int32_t DataSetInStream::ReadDataItem(std::unique_ptr<DataItem>* item) {
  auto data_item = DataSetUtil::Malloc<DataItem>(header()->TensorElemCount());
  return Read(reinterpret_cast<char*>(data_item.get()),
              FlexibleSizeOf(*data_item));
}

}  // namespace oneflow
