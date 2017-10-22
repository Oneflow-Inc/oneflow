#include "oneflow/core/persistence/ubf_header.h"
namespace oneflow {

PersistentOutStream& operator<<(PersistentOutStream& out,
                                const UbfHeader& data) {
  out.Write(reinterpret_cast<const char*>(&data), sizeof(UbfHeader));
  return out;
}

UbfHeader::UbfHeader(const std::string& type, uint32_t data_item_count,
                     const std::vector<uint32_t>& dim_array) {
  CHECK(type.size() <= sizeof(type_));
  type.copy(type_, type.size(), 0);
  data_item_count_ = data_item_count;
  CHECK(dim_array.size() <= sizeof(dim_array_));
  dim_array_size_ = dim_array.size();
  memset(dim_array_, 0, sizeof(dim_array_));
  for (int i = 0; i < dim_array.size(); ++i) { dim_array_[i] = dim_array[i]; }
  UpdateCheckSum();
}

void UbfHeader::UpdateCheckSum() {
  uint32_t chk_sum = GetCheckSum();
  chk_sum -= check_sum_;
  check_sum_ = -chk_sum;
}

uint32_t UbfHeader::GetCheckSum() const {
  static_assert(!(sizeof(*this) % sizeof(uint32_t)), "no alignment");
  uint32_t chk_sum = 0;
  int len = sizeof(*this) / sizeof(uint32_t);
  for (int i = 0; i < len; ++i) {
    chk_sum += reinterpret_cast<const uint32_t*>(this)[i];
  }
  return chk_sum;
}

}  // namespace oneflow
