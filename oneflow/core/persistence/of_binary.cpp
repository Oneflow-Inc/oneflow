#include "oneflow/core/persistence/of_binary.h"

namespace oneflow {

#define DATA_SET_OVERRITE_PERSISTENCE(type)                                \
  PersistentOutStream& operator<<(PersistentOutStream& out,                \
                                  const type& data) {                      \
    out.Write(reinterpret_cast<const char*>(&data), FlexibleSizeOf(data)); \
    return out;                                                            \
  }
OF_PP_FOR_EACH_TUPLE(DATA_SET_OVERRITE_PERSISTENCE, DATA_SET_FORMAT_SEQ);

#define DATA_SET_OVERRITE_OFSTREAM(type)                                   \
  std::ofstream& operator<<(std::ofstream& out, const type& data) {        \
    out.write(reinterpret_cast<const char*>(&data), FlexibleSizeOf(data)); \
    return out;                                                            \
  }
OF_PP_FOR_EACH_TUPLE(DATA_SET_OVERRITE_OFSTREAM, DATA_SET_FORMAT_SEQ);

std::string OfbItem::GetDataId() const {
  return std::string(key_buffer(), key_buffer_len());
}

template<typename T>
void OfbItem::Decode(const Shape& shape, T* out_dptr) {
  switch (data_encode_type_) {
#define OFB_ITEM_DECODE_ENTRY(encode_type)                               \
  case DataEncodeType::encode_type:                                      \
    return OfbDecoder<DataEncodeType::encode_type>::Decode(*this, shape, \
                                                           out_dptr);
    OF_PP_FOR_EACH_TUPLE(OFB_ITEM_DECODE_ENTRY, DATA_ENCODE_TYPE_SEQ)
    default: UNEXPECTED_RUN();
  }
}

namespace {

//  it's only usefull for compiling
void SepcializeTemplate() {
#define SPECIALIZE_OFB_ITEM_DECODE(type, type_case)                     \
  static_cast<OfbItem*>(nullptr)->Decode(*static_cast<Shape*>(nullptr), \
                                         static_cast<type*>(nullptr));
  OF_PP_FOR_EACH_TUPLE(SPECIALIZE_OFB_ITEM_DECODE, ALL_DATA_TYPE_SEQ)
}

}  // namespace

}  // namespace oneflow
