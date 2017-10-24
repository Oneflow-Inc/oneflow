#include "oneflow/core/persistence/ubf_in_stream.h"
#include "oneflow/core/persistence/ubf_util.h"

namespace oneflow {

void UbfInStream::ResetHeader() {
  int ret = in_stream_->Read(reinterpret_cast<char*>(header_.get()),
                             sizeof(UbfHeader));
  CHECK(!ret);
  CHECK(header()->ValidateMagicCode());
  CHECK(!header()->ComputeCheckSum());
}

int32_t UbfInStream::ReadOneItem(
    std::unique_ptr<UbfItem, decltype(&free)>* ubf_item) {
  auto buffer_meta = UbfItem::NewEmpty();
  int ret = ReadMeta(reinterpret_cast<char*>(buffer_meta.get()),
                     Flexible<UbfItem>::SizeOf(*buffer_meta));
  if (ret < 0) { return ret; }
  CHECK(!buffer_meta->ComputeMetaCheckSum());
  *ubf_item = Flexible<UbfItem>::Malloc(buffer_meta->len());
  memcpy(reinterpret_cast<char*>((*ubf_item).get()),
         reinterpret_cast<char*>(buffer_meta.get()),
         Flexible<UbfItem>::SizeOf(*buffer_meta));
  ret = in_stream_->Read((*ubf_item)->mut_data(), (*ubf_item)->len());
  CHECK(!ret);
  CHECK(!(*ubf_item)->ComputeMetaCheckSum());
  return ret;
}

}  // namespace oneflow
