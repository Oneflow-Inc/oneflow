#include "oneflow/core/persistence/ofb_in_stream.h"
#include "oneflow/core/persistence/data_set_util.h"
namespace oneflow {

void OfbInStream::ResetHeader() {
  int ret = in_stream_->Read(reinterpret_cast<char*>(header_.get()),
                             sizeof(OfbHeader));
  CHECK(!ret);
  CHECK(header()->magic_code_ == 0xfeed);
  CHECK(!DataSetUtil::ValidateHeader(*header()));
}

int32_t OfbInStream::ReadOfbItem(
    std::unique_ptr<OfbItem, decltype(&free)>* ofb_item) {
  auto buffer_meta = FlexibleMalloc<OfbItem>(0);
  int ret = ReadMeta(reinterpret_cast<char*>(buffer_meta.get()),
                     FlexibleSizeOf<OfbItem>(0));
  if (ret < 0) { return ret; }
  CHECK(!DataSetUtil::ValidateOfbItem(*buffer_meta));
  *ofb_item = FlexibleMalloc<OfbItem>(buffer_meta->len_);
  memcpy(reinterpret_cast<char*>((*ofb_item).get()),
         reinterpret_cast<char*>(buffer_meta.get()),
         FlexibleSizeOf<OfbItem>(0));
  ret = in_stream_->Read((*ofb_item)->data_, (*ofb_item)->len_);
  CHECK(!ret);
  CHECK(!DataSetUtil::ValidateOfbItem(**ofb_item));
  return ret;
}

}  // namespace oneflow
