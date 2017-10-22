#include "oneflow/core/persistence/ubf_item.h"

namespace oneflow {

PersistentOutStream& operator<<(PersistentOutStream& out, const UbfItem& data) {
  out.Write(reinterpret_cast<const char*>(&data),
            Flexible<UbfItem>::SizeOf(data));
  return out;
}

std::unique_ptr<UbfItem, decltype(&free)> UbfItem::New(
    const std::string& key, size_t value_buf_len, DataType dtype,
    DataEncodeType detype, const std::function<void(char* buff)>& Fill) {
  size_t value_offset = RoundUpToAlignment(key.size(), 8);
  auto ubf_item = Flexible<UbfItem>::Malloc(value_buf_len + value_offset);
  ubf_item->data_type_ = dtype;
  ubf_item->data_encode_type_ = detype;
  ubf_item->key_len_ = key.size();
  ubf_item->value_offset_ = value_offset;
  memset(ubf_item->data_, 0, value_offset);
  key.copy(ubf_item->mut_key_buffer(), key.size());
  if (value_buf_len) { Fill(const_cast<char*>(ubf_item->mut_value_buffer())); }
  ubf_item->UpdateCheckSum();
  return ubf_item;
}

uint8_t UbfItem::GetMetaCheckSum() const {
  uint8_t chk_sum = 0;
  int meta_len = Flexible<UbfItem>::SizeOf(0);
  for (int i = 0; i < meta_len; ++i) {
    chk_sum += reinterpret_cast<const char*>(this)[i];
  }
  return chk_sum;
}

void UbfItem::UpdateMetaCheckSum() {
  uint8_t meta_chk_sum = GetMetaCheckSum();
  meta_chk_sum -= meta_check_sum_;
  meta_check_sum_ = -meta_chk_sum;
}

uint8_t UbfItem::GetDataCheckSum() const {
  uint8_t data_chk_sum = 0;
  for (int i = 0; i < len_; ++i) { data_chk_sum += data_[i]; }
  return data_chk_sum;
}

void UbfItem::UpdateCheckSum() {
  UpdateDataCheckSum();
  UpdateMetaCheckSum();
}

void UbfItem::UpdateDataCheckSum() {
  uint8_t data_chk_sum = GetDataCheckSum();
  data_check_sum_ = -data_chk_sum;
}

std::string UbfItem::GetDataId() const {
  return std::string(key_buffer(), key_buffer_len());
}

template<typename T>
void UbfItem::Decode(const Shape& shape, T* out_dptr) {
  switch (data_encode_type_) {
#define UBF_ITEM_DECODE_ENTRY(encode_type)                               \
  case DataEncodeType::encode_type:                                      \
    return UbfDecoder<DataEncodeType::encode_type>::Decode(*this, shape, \
                                                           out_dptr);
    OF_PP_FOR_EACH_TUPLE(UBF_ITEM_DECODE_ENTRY, DATA_ENCODE_TYPE_SEQ)
    default: UNEXPECTED_RUN();
  }
}

namespace {

//  it's only usefull for compiling
void SepcializeTemplate() {
#define SPECIALIZE_UBF_ITEM_DECODE(type, type_case)                     \
  static_cast<UbfItem*>(nullptr)->Decode(*static_cast<Shape*>(nullptr), \
                                         static_cast<type*>(nullptr));
  OF_PP_FOR_EACH_TUPLE(SPECIALIZE_UBF_ITEM_DECODE, ALL_DATA_TYPE_SEQ)
}

}  // namespace

}  // namespace oneflow
