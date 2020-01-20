#ifndef ONEFLOW_CORE_COMMON_OBJECT_MSG_H_
#define ONEFLOW_CORE_COMMON_OBJECT_MSG_H_

#include <atomic>
#include <type_traits>
#include "oneflow/core/common/struct_traits.h"

namespace oneflow {

#define BEGIN_OBJECT_MSG(class_name)                                 \
  class OBJECT_MSG_TYPE(class_name) final : public ObjectMsgStruct { \
   public:                                                           \
    using self_type = OBJECT_MSG_TYPE(class_name);                   \
    BEGIN_DSS(sizeof(ObjectMsgStruct));

#define END_OBJECT_MSG(class_name)                    \
  END_DSS("object_msg", OBJECT_MSG_TYPE(class_name)); \
  }                                                   \
  ;

#define OBJECT_MSG_DEFINE_FIELD(field_type, field_name) \
  _OBJECT_MSG_DEFINE_FIELD(field_type, field_name)      \
  DSS_DEFINE_FIELD("object_msg", self_type, OF_PP_CAT(field_name, _));

#define OBJECT_MSG_DEFINE_RAW_PTR_FIELD(field_type, field_name) \
  _OBJECT_MSG_DEFINE_RAW_POINTER_FIELD(field_type, field_name)  \
  DSS_DEFINE_FIELD("object_msg", self_type, OF_PP_CAT(field_name, _));

#define OBJECT_MSG_PTR(class_name) ObjectMsgPtr<OBJECT_MSG_TYPE(class_name)>

#define OBJECT_MSG_TYPE(class_name) OF_PP_CAT(class_name, __object_msg_struct_type__)

// details

#define OBJECT_MSG_STRUCT_MEMBER(class_name)                           \
  std::conditional<ObjMsgIsScalar<OBJECT_MSG_TYPE(class_name)>::value, \
                   OBJECT_MSG_TYPE(class_name), OBJECT_MSG_TYPE(class_name)*>::type

typedef char OBJECT_MSG_TYPE(char);
typedef int8_t OBJECT_MSG_TYPE(int8_t);
typedef uint8_t OBJECT_MSG_TYPE(uint8_t);
typedef int16_t OBJECT_MSG_TYPE(int16_t);
typedef uint16_t OBJECT_MSG_TYPE(uint16_t);
typedef int32_t OBJECT_MSG_TYPE(int32_t);
typedef uint32_t OBJECT_MSG_TYPE(uint32_t);
typedef int64_t OBJECT_MSG_TYPE(int64_t);
typedef uint64_t OBJECT_MSG_TYPE(uint64_t);
typedef float OBJECT_MSG_TYPE(float);
typedef double OBJECT_MSG_TYPE(double);
typedef std::string OBJECT_MSG_TYPE(string);

#define _OBJECT_MSG_DEFINE_FIELD(field_type, field_name)                                       \
 public:                                                                                       \
  using OF_PP_CAT(field_name, __field_type__) = typename OBJECT_MSG_STRUCT_MEMBER(field_type); \
  DEFINE_SETTER(ObjMsgIsScalar, OF_PP_CAT(field_name, __field_type__), field_name);            \
  DEFINE_GETTER(OF_PP_CAT(field_name, __field_type__), field_name);                            \
  DEFINE_MUTABLE(OF_PP_CAT(field_name, __field_type__), field_name);                           \
                                                                                               \
 private:                                                                                      \
  OF_PP_CAT(field_name, __field_type__) OF_PP_CAT(field_name, _);

#define _OBJECT_MSG_DEFINE_RAW_POINTER_FIELD(field_type, field_name) \
 public:                                                             \
  static_assert(std::is_pointer<field_type>::value,                  \
                OF_PP_STRINGIZE(field_type) "is not a pointer");     \
  DEFINE_SETTER(ObjMsgIsScalar, field_type, field_name);             \
  DEFINE_GETTER(field_type, field_name);                             \
  DEFINE_MUTABLE(field_type, field_name);                            \
                                                                     \
 private:                                                            \
  field_type OF_PP_CAT(field_name, _);

class ObjectMsgPtrBaseUtil;

class ObjectMsgStruct {
 public:
  void __Delete__() {}

 private:
  friend class ObjectMsgPtrBaseUtil;
  void __InitRefCount__() { ref_cnt_ = 0; }
  void __IncreaseRefCount__() { ref_cnt_++; }
  int32_t __DecreaseRefCount__() { return --ref_cnt_; }

  std::atomic<int32_t> ref_cnt_;
};

class ObjectMsgPtrBaseUtil {
 protected:
  static void IncreaseRefCount(ObjectMsgStruct* ptr) { ptr->__IncreaseRefCount__(); }
  static int32_t DecreaseRefCount(ObjectMsgStruct* ptr) { return ptr->__DecreaseRefCount__(); }
};

template<typename T>
class ObjectMsgPtrUtil : private ObjectMsgPtrBaseUtil {
 public:
  static void Ref(T* ptr) { IncreaseRefCount(ptr); }
  static void Release(T* ptr) {
    if (ptr == nullptr) { return; }
    if (DecreaseRefCount(ptr) > 0) { return; }
    ptr->__Delete__();
    delete ptr;
  }
};

template<typename T>
class ObjectMsgPtr final {
 public:
  ~ObjectMsgPtr() { ObjectMsgPtrUtil<T>::Release(ptr_); }

  static ObjectMsgPtr New() {
    auto* ptr = new T();
    std::memset(reinterpret_cast<void*>(ptr), 0, sizeof(T));
    return ObjectMsgPtr(ptr);
  }

  ObjectMsgPtr& operator=(const ObjectMsgPtr& rhs) {
    ObjectMsgPtrUtil<T>::Release(ptr_);
    ptr_ = rhs.ptr_;
    ObjectMsgPtrUtil<T>::Ref(ptr_);
    return *this;
  }

  operator bool() const { return ptr_ == nullptr; }
  T* get() const { return ptr_; }
  T* operator->() const { return ptr_; }
  const T& operator*() const { return *ptr_; }

 private:
  ObjectMsgPtr() : ptr_(nullptr) {}
  ObjectMsgPtr(T* ptr) {
    static_assert(std::is_base_of<ObjectMsgStruct, T>::value,
                  "T is not derived from ObjectMsgStruct");
    ptr_ = ptr;
    ObjectMsgPtrUtil<T>::Ref(ptr);
  }
  T* ptr_;
};

template<typename T>
struct ObjMsgIsScalar {
  const static bool value =
      std::is_arithmetic<T>::value || std::is_enum<T>::value || std::is_same<T, std::string>::value;
};
}

#endif  // ONEFLOW_CORE_COMMON_OBJECT_MSG_H_
