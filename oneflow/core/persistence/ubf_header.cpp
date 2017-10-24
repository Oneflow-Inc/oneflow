#include "oneflow/core/persistence/ubf_header.h"

namespace oneflow {

PersistentOutStream& operator<<(PersistentOutStream& out,
                                const UbfHeader& data) {
  out.Write(reinterpret_cast<const char*>(&data), sizeof(UbfHeader));
  return out;
}

UbfHeader::UbfHeader(uint32_t ubf_item_num,
                     const std::vector<uint32_t>& dim_array) {
  ubf_item_num_ = ubf_item_num;
  CHECK(dim_array.size() <= sizeof(dim_array_));
  dim_array_size_ = dim_array.size();
  memset(dim_array_, 0, sizeof(dim_array_));
  for (int i = 0; i < dim_array.size(); ++i) { dim_array_[i] = dim_array[i]; }
}

}  // namespace oneflow
