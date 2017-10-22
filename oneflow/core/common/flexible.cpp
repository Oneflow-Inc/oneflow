#include "oneflow/core/common/flexible.h"
#include "oneflow/core/common/preprocessor.h"
#include "oneflow/core/persistence/ubf_item.h"

#define FLAXIBLE_STRUCT_SEQ OF_PP_MAKE_TUPLE_SEQ(UbfItem, len_, data_)

namespace oneflow {

#define DEFINE_FLEXIBLE_SIZE_OF(type, len, array)                         \
  template<>                                                              \
  size_t Flexible<type>::SizeOf(size_t n) {                               \
    type* ptr = nullptr;                                                  \
    return sizeof(type) - sizeof(ptr->array) + n * sizeof(ptr->array[0]); \
  }
OF_PP_FOR_EACH_TUPLE(DEFINE_FLEXIBLE_SIZE_OF, FLAXIBLE_STRUCT_SEQ);

#define SPEC_FLEXIBLE_SET_ARRAY_SIZE(type, len, array)     \
  template<>                                               \
  void Flexible<type>::SetArraySize(type* obj, size_t l) { \
    obj->len = l;                                          \
  }
OF_PP_FOR_EACH_TUPLE(SPEC_FLEXIBLE_SET_ARRAY_SIZE, FLAXIBLE_STRUCT_SEQ);

#define DEFINE_FLEXIBLE_OBJ_SIZE_OF(type, len, array) \
  template<>                                          \
  size_t Flexible<type>::SizeOf(const type& obj) {    \
    return Flexible<type>::SizeOf(obj.len);           \
  }
OF_PP_FOR_EACH_TUPLE(DEFINE_FLEXIBLE_OBJ_SIZE_OF, FLAXIBLE_STRUCT_SEQ);

template<typename T>
std::unique_ptr<T, decltype(&free)> Flexible<T>::Malloc(size_t len) {
  T* ptr = reinterpret_cast<T*>(malloc(Flexible<T>::SizeOf(len)));
  Flexible<T>::SetArraySize(ptr, len);
  return std::unique_ptr<T, decltype(&free)>(ptr, &free);
}

namespace {

//  only useful for compiling
void SpecializeTemplate() {
#define DEFINE_FLEXIBLE_MALLOC(type, len, array) Flexible<type>::Malloc(0);
  OF_PP_FOR_EACH_TUPLE(DEFINE_FLEXIBLE_MALLOC, FLAXIBLE_STRUCT_SEQ);
}

}  // namespace
}
