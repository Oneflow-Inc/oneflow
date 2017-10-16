#include "oneflow/core/persistence/data_set_in_stream.h"
#include "oneflow/core/persistence/data_set_util.h"
namespace oneflow {

void DataSetInStream::InitHeader() {
  CHECK(file_size());
  CHECK(!cur_file_pos());
  Read(reinterpret_cast<char*>(header_.get()), sizeof(DataSetHeader));
  CHECK(header()->magic_code == 0xfeed);
  size_t data_body_size =
      header()->data_item_count
      * FlexibleSizeOf<DataItem>(header()->TensorElemCount());
  CHECK(file_size() == header()->DataBodyOffset() + data_body_size);
}

void DataSetInStream::InitItemDesc() {
  auto desc = DataSetUtil::Malloc<DataItemDesc>(header()->data_item_count);
  CHECK(!Read(reinterpret_cast<char*>(desc.get()), FlexibleSizeOf(*desc)));
  item_desc_ = std::move(desc);
  CHECK(item_desc_->data_item_count == header()->data_item_count);
}

void DataSetInStream::InitLabelDesc() {
  if (header()->label_desc_buf_len) {
    size_t buf_len = header()->label_desc_buf_len;
    char* buf = reinterpret_cast<char*>(malloc(buf_len));
    Read(buf, buf_len);
    std::unique_ptr<DataSetLabelDesc, decltype(&free)> label_desc(
        reinterpret_cast<DataSetLabelDesc*>(buf), &free);
    label_desc_ = std::move(label_desc);
    CHECK(FlexibleSizeOf<DataSetLabelDesc>(label_desc_->label_array_size)
          == buf_len);
  }
}

int32_t DataSetInStream::ReadDataItem(
    std::unique_ptr<DataItem, decltype(&free)>* data_item) {
  size_t buf_len = item_desc()->data_item_buf_len[cur_item_pos()];
  char* buf = reinterpret_cast<char*>(malloc(buf_len));
  *data_item = std::unique_ptr<DataItem, decltype(&free)>(
      reinterpret_cast<DataItem*>(buf), &free);
  int32_t ret = Read(buf, buf_len);
  CHECK(FlexibleSizeOf(*(*data_item)) == buf_len);
  return ret;
}

}  // namespace oneflow
