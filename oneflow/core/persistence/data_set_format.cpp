#include "oneflow/core/persistence/data_set_format.h"

namespace oneflow {

#define DEFINE_FLEXIBLE_SIZE_OF(type, len, array)        \
  template<>                                             \
  size_t FlexibleSizeOf<type>(uint32_t n) {              \
    type* ptr = nullptr;                                 \
    return sizeof(ptr->len) + n * sizeof(ptr->array[0]); \
  }
OF_PP_FOR_EACH_TUPLE(DEFINE_FLEXIBLE_SIZE_OF, FLAXIBLE_STRUCT_SEQ);

#define DEFINE_FLEXIBLE_OBJ_SIZE_OF(type, len, array) \
  template<>                                          \
  size_t FlexibleSizeOf<type>(const type& obj) {      \
    return FlexibleSizeOf<type>(obj.len);             \
  }
OF_PP_FOR_EACH_TUPLE(DEFINE_FLEXIBLE_OBJ_SIZE_OF, FLAXIBLE_STRUCT_SEQ);

template<>
size_t FlexibleSizeOf<DataSetHeader>(const DataSetHeader& obj) {
  return sizeof(DataSetHeader);
}

#define DATA_SET_OVERRITE_OFSTREAM(type)                                   \
  std::ostream& operator<<(std::ostream& out, const type& data) {          \
    out.write(reinterpret_cast<const char*>(&data), FlexibleSizeOf(data)); \
    return out;                                                            \
  }
OF_PP_FOR_EACH_TUPLE(DATA_SET_OVERRITE_OFSTREAM, DATA_SET_FORMAT_SEQ);

}  // namespace oneflow
