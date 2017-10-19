#include "oneflow/core/persistence/data_set_format.h"

namespace oneflow {

#define DEFINE_FLEXIBLE_SIZE_OF(type, len, array)                         \
  template<>                                                              \
  size_t FlexibleSizeOf<type>(uint32_t n) {                               \
    type* ptr = nullptr;                                                  \
    return sizeof(type) - sizeof(ptr->array) + n * sizeof(ptr->array[0]); \
  }
OF_PP_FOR_EACH_TUPLE(DEFINE_FLEXIBLE_SIZE_OF, FLAXIBLE_STRUCT_SEQ);

#define SPEC_FLEXIBLE_SET_ARRAY_SIZE(type, len, array)    \
  template<>                                              \
  void FlexibleSetArraySize<type>(type * obj, size_t l) { \
    obj->len = l;                                         \
  }
OF_PP_FOR_EACH_TUPLE(SPEC_FLEXIBLE_SET_ARRAY_SIZE, FLAXIBLE_STRUCT_SEQ);

#define DEFINE_FLEXIBLE_OBJ_SIZE_OF(type, len, array) \
  template<>                                          \
  size_t FlexibleSizeOf<type>(const type& obj) {      \
    return FlexibleSizeOf<type>(obj.len);             \
  }
OF_PP_FOR_EACH_TUPLE(DEFINE_FLEXIBLE_OBJ_SIZE_OF, FLAXIBLE_STRUCT_SEQ);

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

std::string Record::GetKey() const {
  return std::string(key_buffer(), key_buffer_len());
}

template<typename T>
void Record::Decode(const Shape& shape, T* out_dptr) {
  switch (data_encode_type_) {
#define RECORD_DECODE_ENTRY(encode_type)                                       \
  case DataEncodeType::encode_type:                                            \
    return RecordDecoder<DataEncodeType::encode_type>::Decode<T>(*this, shape, \
                                                                 out_dptr);
    OF_PP_FOR_EACH_TUPLE(RECORD_DECODE_ENTRY, DATA_ENCODE_TYPE_SEQ)
    default: UNEXPECTED_RUN();
  }
}

namespace {

//  it's only usefull for compiling
void SepcializeTemplate() {
#define SPECIALIZE_RECORD_DECODE(type, type_case)                      \
  static_cast<Record*>(nullptr)->Decode(*static_cast<Shape*>(nullptr), \
                                        static_cast<type*>(nullptr));
  OF_PP_FOR_EACH_TUPLE(SPECIALIZE_RECORD_DECODE, ALL_DATA_TYPE_SEQ)
}

}  // namespace

}  // namespace oneflow
