#include "oneflow/core/persistence/ubf_in_stream.h"
#include "oneflow/core/persistence/ubf_util.h"

namespace oneflow {

int32_t UbfInStream::ReadOneItem(std::unique_ptr<UbfItem>* ubf_item) {
  auto desc = of_make_unique<UbfItemDesc>();
  int ret = ReadDesc(reinterpret_cast<char*>(desc.get()), sizeof(*desc));
  if (ret < 0) { return ret; }
  *ubf_item = of_make_unique<UbfItem>(std::move(desc));
  ret = in_stream_->Read((*ubf_item)->mut_data(), (*ubf_item)->len());
  CHECK(!ret);
  return ret;
}

}  // namespace oneflow
