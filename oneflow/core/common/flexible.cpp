#include "oneflow/core/common/flexible.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/persistence/of_binary.h"

#define FLAXIBLE_STRUCT_SEQ OF_PP_MAKE_TUPLE_SEQ(OfbItem, len_, data_)

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
}
