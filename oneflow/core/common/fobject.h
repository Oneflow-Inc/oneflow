#ifndef ONEFLOW_CORE_COMMON_FOBJECT_H_
#define ONEFLOW_CORE_COMMON_FOBJECT_H_

#include <atomic>
#include <type_traits>
#include "oneflow/core/common/struct_traits.h"

namespace oneflow {

#define BEGIN_FOBJECT(class_name)                               \
  class FOBJECT_TYPE(class_name) final : public FObjectStruct { \
   public:                                                      \
    using self_type = FOBJECT_TYPE(class_name);                 \
    DSS_DECLARE_CODE_LINE_FIELD_SIZE_AND_OFFSET(sizeof(FObjectStruct));

#define END_FOBJECT(class_name) \
  }                             \
  ;

//  DSS_STATIC_ASSERT_STRUCT_SIZE("fobject", FOBJECT_TYPE(class_name));

#define FOBJECT_DEFINE_FIELD(field_type, field_name) \
  _FOBJECT_DEFINE_FIELD(field_type, field_name)      \
  DSS_DEFINE_AND_CHECK_CODE_LINE_FIELD("fobject", self_type, OF_PP_CAT(field_name, _));

#define FOBJECT_METHOD(class_name, method_name) FOBJECT_TYPE(class_name)::method_name

#define FOBJECT_STRUCT_MEMBER(class_name)                                \
  std::conditional<std::is_arithmetic<FOBJECT_TYPE(class_name)>::value   \
                       || std::is_enum<FOBJECT_TYPE(class_name)>::value, \
                   FOBJECT_TYPE(class_name), FObjectStructMember<FOBJECT_TYPE(class_name)>>::type

#define FOBJECT(class_name)                                              \
  std::conditional<std::is_arithmetic<FOBJECT_TYPE(class_name)>::value   \
                       || std::is_enum<FOBJECT_TYPE(class_name)>::value, \
                   FOBJECT_TYPE(class_name), FObject<FOBJECT_TYPE(class_name)>>::type

#define FOBJECT_TYPE(class_name) OF_PP_CAT(class_name, __fobject_struct_type__)

// details

#define DEFINE_FOBJECT_BASIC_TYPE(class_name) typedef class_name FOBJECT_TYPE(class_name)

DEFINE_FOBJECT_BASIC_TYPE(char);
DEFINE_FOBJECT_BASIC_TYPE(int8_t);
DEFINE_FOBJECT_BASIC_TYPE(uint8_t);
DEFINE_FOBJECT_BASIC_TYPE(int16_t);
DEFINE_FOBJECT_BASIC_TYPE(uint16_t);
DEFINE_FOBJECT_BASIC_TYPE(int32_t);
DEFINE_FOBJECT_BASIC_TYPE(uint32_t);
DEFINE_FOBJECT_BASIC_TYPE(int64_t);
DEFINE_FOBJECT_BASIC_TYPE(uint64_t);
DEFINE_FOBJECT_BASIC_TYPE(float);
DEFINE_FOBJECT_BASIC_TYPE(double);

#define _FOBJECT_DEFINE_FIELD(field_type, field_name)            \
 public:                                                         \
  const FOBJECT_STRUCT_MEMBER(field_type) & field_name() const { \
    return OF_PP_CAT(field_name, _);                             \
  }                                                              \
                                                                 \
 private:                                                        \
  template<typename T>                                           \
  void OF_PP_CAT(set_, field_name)(const T& fobj) {              \
    OF_PP_CAT(field_name, _) = fobj;                             \
  }                                                              \
  FOBJECT_STRUCT_MEMBER(field_type) OF_PP_CAT(field_name, _);

class FObjectBase;

class FObjectStruct {
 public:
  void __Init__() {}
  void __Delete__() {}

 private:
  friend class FObjectBase;
  void __InitRefCount__() { ref_cnt_ = 0; }
  void __IncreaseRefCount__() { ref_cnt_++; }
  int32_t __DecreaseRefCount__() { return --ref_cnt_; }

  std::atomic<int32_t> ref_cnt_;
};

class FObjectBase {
 protected:
  FObjectBase() = default;
  ~FObjectBase() = default;

  void IncreaseRefCount(FObjectStruct* ptr) { ptr->__IncreaseRefCount__(); }
  int32_t DecreaseRefCount(FObjectStruct* ptr) { return ptr->__DecreaseRefCount__(); }
};

template<typename T>
class FObjectStructPtr : private FObjectBase {
 public:
  operator bool() const { return ptr_ == nullptr; }
  T* get() const { return ptr_; }
  T* operator->() const { return ptr_; }
  const T& operator*() const { return *ptr_; }

 protected:
  void Ref(T* ptr) {
    ptr_ = ptr;
    IncreaseRefCount(ptr_);
  }
  void Release() {
    if (ptr_ != nullptr && DecreaseRefCount(ptr_) == 0) {
      ptr_->__Delete__();
      delete ptr_;
    }
    ptr_ = nullptr;
  }

  T* ptr_;
};
template<typename T>
class FObject;

template<typename T>
class FObjectStructMember final : public FObjectStructPtr<T> {
 public:
  FObjectStructMember<T>& operator=(const FObject<T>& rhs);
  FObjectStructMember<T>& operator=(const FObjectStructMember<T>& rhs) {
    this->Ref(rhs.ptr_);
    return *this;
  }
  friend class FObject<T>;
};

template<typename T>
class FObject final : public FObjectStructPtr<T> {
 public:
  FObject(const FObjectStructMember<T>& ptr) { this->Ref(ptr); }
  ~FObject() { this->Release(); }

  template<typename... Args>
  static FObject New(Args&&... args) {
    auto* ptr = new T();
    ptr->__Init__(std::forward<Args>(args)...);
    return FObject(ptr);
  }

  FObject& operator=(const FObject& rhs) {
    this->Ref(rhs.ptr_);
    return *this;
  }
  FObject& operator=(const FObjectStructMember<T>& rhs) {
    this->Ref(rhs.ptr_);
    return *this;
  }

 private:
  friend class FObjectStructMember<T>;
  FObject(T* ptr) { this->Ref(ptr); }
};

template<typename T>
FObjectStructMember<T>& FObjectStructMember<T>::operator=(const FObject<T>& rhs) {
  this->Ref(rhs.ptr_);
  return *this;
}
}

#endif  // ONEFLOW_CORE_COMMON_FOBJECT_H_
