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

#define DATA_SET_OVERRITE_OFSTREAM(type)                                   \
  std::ostream& operator<<(std::ostream& out, const type& data) {          \
    out.write(reinterpret_cast<const char*>(&data), FlexibleSizeOf(data)); \
    return out;                                                            \
  }
OF_PP_FOR_EACH_TUPLE(DATA_SET_OVERRITE_OFSTREAM, DATA_SET_FORMAT_SEQ);

size_t DataSetHeader::TensorElemCount() const {
  int count = 1;
  for (int i = 0; i < dim_array_size; i++) { count *= dim_array[i]; }
  return count;
}
size_t DataSetHeader::DataBodyOffset() const { return sizeof(*this); }

}  // namespace oneflow
