#ifndef ONEFLOW_CORE_COMMON_OBJECT_MSG_H_
#define ONEFLOW_CORE_COMMON_OBJECT_MSG_H_

#include <atomic>
#include <memory>
#include <type_traits>
#include "oneflow/core/common/struct_traits.h"

namespace oneflow {

#define BEGIN_OBJECT_MSG(class_name)                                                           \
  class OBJECT_MSG_TYPE(class_name) final : public ObjectMsgStruct {                           \
   public:                                                                                     \
    BEGIN_DSS(DSS_GET_DEFINE_COUNTER(), OBJECT_MSG_TYPE(class_name), sizeof(ObjectMsgStruct)); \
    OBJECT_MSG_DEFINE_DEFAULT(OBJECT_MSG_TYPE(class_name));

#define END_OBJECT_MSG(class_name)                                              \
  END_DSS(DSS_GET_DEFINE_COUNTER(), "object_msg", OBJECT_MSG_TYPE(class_name)); \
  }                                                                             \
  ;

#define OBJECT_MSG_DEFINE_FIELD(field_type, field_name) \
  _OBJECT_MSG_DEFINE_FIELD(field_type, field_name)      \
  DSS_DEFINE_FIELD(DSS_GET_DEFINE_COUNTER(), "object_msg", OF_PP_CAT(field_name, _));

#define OBJECT_MSG_DEFINE_RAW_PTR_FIELD(field_type, field_name) \
  _OBJECT_MSG_DEFINE_RAW_POINTER_FIELD(field_type, field_name)  \
  DSS_DEFINE_FIELD(DSS_GET_DEFINE_COUNTER(), "object_msg", OF_PP_CAT(field_name, _));

#define OBJECT_MSG_DEFINE_ONEOF_FIELD(oneof_name, type_and_field_name_seq)       \
  OF_PP_FOR_EACH_TUPLE(OBJECT_MSG_TYPEDEF_ONEOF_FIELD, type_and_field_name_seq); \
  OBJECT_MSG_DEFINE_ONEOF_ENUM_TYPE(oneof_name, type_and_field_name_seq);        \
  OBJECT_MSG_DEFINE_ONEOF_UNION(oneof_name, type_and_field_name_seq);            \
  OBJECT_MSG_DEFINE_ONEOF_ACCESSOR(oneof_name, type_and_field_name_seq);         \
  OBJECT_MSG_DSS_DEFINE_UION_FIELD(DSS_GET_DEFINE_COUNTER(), oneof_name, type_and_field_name_seq);

#define OBJECT_MSG_ONEOF_FIELD(field_type, field_name) \
  OF_PP_MAKE_TUPLE_SEQ(OBJECT_MSG_TYPE(field_type), field_name)

#define OBJECT_MSG_PTR(class_name) ObjectMsgPtr<OBJECT_MSG_TYPE(class_name)>

#define OBJECT_MSG_TYPE(class_name) OF_PP_CAT(class_name, __object_msg_struct_type__)

// details

#define OBJECT_MSG_TYPEDEF_ONEOF_FIELD(field_type, field_name) \
  using OF_PP_CAT(field_name, _OneofFieldType) = typename _OBJECT_MSG_STRUCT_MEMBER(field_type);

#define OBJECT_MSG_DSS_DEFINE_UION_FIELD(define_counter, oneof_name, type_and_field_name_seq) \
  DSS_DEFINE_FIELD(define_counter, "object message", OF_PP_CAT(oneof_name, _));               \
  DSS_DEFINE_UNION_FIELD_VISITOR(                                                             \
      define_counter, case_,                                                                  \
      OF_PP_FOR_EACH_TUPLE(OBJECT_MSG_MAKE_UNION_TYPE7FIELD4CASE, type_and_field_name_seq));

#define OBJECT_MSG_MAKE_UNION_TYPE7FIELD4CASE(field_type, field_name)                    \
  OF_PP_MAKE_TUPLE_SEQ(OF_PP_CAT(field_name, _OneofFieldType), OF_PP_CAT(field_name, _), \
                       _OBJECT_MSG_ONEOF_ENUM_VALUE(field_name))

#define OBJECT_MSG_DEFINE_ONEOF_ACCESSOR(oneof_name, type_and_field_name_seq)                      \
  _OBJECT_MSG_DEFINE_ONEOF_CASE_ACCESSOR(oneof_name, _OBJECT_MSG_ONEOF_ENUM_TYPE(oneof_name));     \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_OBJECT_MSG_ONEOF_ACCESSOR, (_OBJECT_MSG_ONEOF_ENUM_VALUE), \
                                   (oneof_name), type_and_field_name_seq)

#define MAKE_OBJECT_MSG_ONEOF_ACCESSOR(get_enum_value, oneof_name, pair)                          \
 public:                                                                                          \
  const OF_PP_PAIR_FIRST(pair) & OF_PP_PAIR_SECOND(pair)() const {                                \
    if (OF_PP_CAT(has_, OF_PP_PAIR_SECOND(pair))()) {                                             \
      return GetterTrait<std::is_base_of<ObjectMsgStruct, OF_PP_PAIR_FIRST(pair)>::value>::Call(  \
          OF_PP_CAT(oneof_name, _).OF_PP_CAT(OF_PP_PAIR_SECOND(pair), _));                        \
    }                                                                                             \
    return ObjectMsgGetDefault<std::is_base_of<ObjectMsgStruct, OF_PP_PAIR_FIRST(pair)>::value>:: \
        Call(&OF_PP_CAT(oneof_name, _).OF_PP_CAT(OF_PP_PAIR_SECOND(pair), _));                    \
  }                                                                                               \
  bool OF_PP_CAT(has_, OF_PP_PAIR_SECOND(pair))() const {                                         \
    return OF_PP_CAT(oneof_name, _case)() == get_enum_value(OF_PP_PAIR_SECOND(pair));             \
  }                                                                                               \
  void OF_PP_CAT(clear_, OF_PP_PAIR_SECOND(pair))() {                                             \
    if (!OF_PP_CAT(has_, OF_PP_PAIR_SECOND(pair))()) { return; }                                  \
    OF_PP_CAT(set_, OF_PP_CAT(oneof_name, _case))                                                 \
    (_OBJECT_MSG_ONEOF_NOT_SET_VALUE(oneof_name));                                                \
  }
//  OF_PP_PAIR_FIRST(pair) * OF_PP_CAT(mutable_, OF_PP_PAIR_SECOND(pair))() {
//    OF_PP_CAT(set_, OF_PP_CAT(oneof_name, _case))
//    (get_enum_value(OF_PP_PAIR_SECOND(pair)));
//    return &OF_PP_CAT(oneof_name, _).OF_PP_CAT(OF_PP_PAIR_SECOND(pair), _);
//  }
//  void OF_PP_CAT(set_, OF_PP_PAIR_SECOND(pair))(const OF_PP_PAIR_FIRST(pair) & val) {
//    *OF_PP_CAT(mutable_, OF_PP_PAIR_SECOND(pair))() = val;
//  }

#define _OBJECT_MSG_DEFINE_ONEOF_CASE_ACCESSOR(oneof_name, T)                       \
 public:                                                                            \
  T OF_PP_CAT(oneof_name, _case)() const { return OF_PP_CAT(oneof_name, _).case_; } \
                                                                                    \
 private:                                                                           \
  void OF_PP_CAT(set_, OF_PP_CAT(oneof_name, _case))(T val) {                       \
    OF_PP_CAT(oneof_name, _).case_ = val;                                           \
  }

#define OBJECT_MSG_DEFINE_ONEOF_UNION(oneof_name, type_and_field_name_seq)             \
 private:                                                                              \
  struct {                                                                             \
    union {                                                                            \
      OF_PP_FOR_EACH_TUPLE(MAKE_OBJECT_MSG_ONEOF_UNION_FIELD, type_and_field_name_seq) \
    };                                                                                 \
    _OBJECT_MSG_ONEOF_ENUM_TYPE(oneof_name) case_;                                     \
  } OF_PP_CAT(oneof_name, _);

#define MAKE_OBJECT_MSG_ONEOF_UNION_FIELD(field_type, field_name) \
  _OBJECT_MSG_STRUCT_MEMBER(field_type) OF_PP_CAT(field_name, _);

#define OBJECT_MSG_DEFINE_ONEOF_ENUM_TYPE(oneof_name, type_and_field_name_seq)     \
 public:                                                                           \
  enum _OBJECT_MSG_ONEOF_ENUM_TYPE(oneof_name) {                                   \
    _OBJECT_MSG_ONEOF_NOT_SET_VALUE(oneof_name) = 0,                               \
    OF_PP_FOR_EACH_TUPLE(MAKE_OBJECT_MSG_ONEOF_ENUM_CASE, type_and_field_name_seq) \
  }

#define MAKE_OBJECT_MSG_ONEOF_ENUM_CASE(field_type, field_name) \
  _OBJECT_MSG_ONEOF_ENUM_VALUE(field_name),

#define _OBJECT_MSG_ONEOF_ENUM_TYPE(oneof_name) OF_PP_CAT(oneof_name, _OneofEnumType)
#define _OBJECT_MSG_ONEOF_ENUM_VALUE(field) OF_PP_CAT(field, _OneofEnumValue)
#define _OBJECT_MSG_ONEOF_NOT_SET_VALUE(oneof_name) OF_PP_CAT(k_, OF_PP_CAT(oneof_name, _NotSet))

#define OBJECT_MSG_STRUCT_MEMBER(class_name) _OBJECT_MSG_STRUCT_MEMBER(OBJECT_MSG_TYPE(class_name))

#define _OBJECT_MSG_STRUCT_MEMBER(class_name) \
  std::conditional<ObjectMsgIsScalar<class_name>::value, class_name, class_name*>::type

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
  DEFINE_SETTER(ObjectMsgIsScalar, OF_PP_CAT(field_name, __field_type__), field_name);         \
  DEFINE_GETTER(OF_PP_CAT(field_name, __field_type__), field_name);                            \
  DEFINE_MUTABLE(OF_PP_CAT(field_name, __field_type__), field_name);                           \
                                                                                               \
 private:                                                                                      \
  OF_PP_CAT(field_name, __field_type__) OF_PP_CAT(field_name, _);

#define _OBJECT_MSG_DEFINE_RAW_POINTER_FIELD(field_type, field_name)                           \
 public:                                                                                       \
  static_assert(std::is_pointer<field_type>::value,                                            \
                OF_PP_STRINGIZE(field_type) "is not a pointer");                               \
  void OF_PP_CAT(set_raw_ptr_, field_name)(field_type val) { OF_PP_CAT(field_name, _) = val; } \
  DEFINE_SETTER(ObjectMsgIsScalar, field_type, field_name);                                    \
  DEFINE_GETTER(field_type, field_name);                                                       \
  DEFINE_MUTABLE(field_type, field_name);                                                      \
                                                                                               \
 private:                                                                                      \
  field_type OF_PP_CAT(field_name, _);

#define OBJECT_MSG_DEFINE_DEFAULT(object_msg_type_name)                           \
  const object_msg_type_name& __Default__() const {                               \
    static const ObjectMsgStructDefault<object_msg_type_name> default_object_msg; \
    return default_object_msg.Get();                                              \
  }

class ObjectMsgAllocator {
 public:
  virtual char* Allocate(std::size_t size) = 0;
  virtual void Deallocate(char* ptr, std::size_t size) = 0;

 protected:
  ObjectMsgAllocator() = default;
};

class ObjectMsgDefaultAllocator : public ObjectMsgAllocator {
 public:
  ObjectMsgDefaultAllocator() = default;

  static ObjectMsgDefaultAllocator* GlobalObjectMsgAllocator() {
    static ObjectMsgDefaultAllocator allocator;
    return &allocator;
  }

  char* Allocate(std::size_t size) override { return allocator_.allocate(size); }
  void Deallocate(char* ptr, std::size_t size) override { allocator_.deallocate(ptr, size); }

 private:
  std::allocator<char> allocator_;
};

class ObjectMsgPtrUtil;
template<typename T>
class ObjectMsgPtr;

class ObjectMsgStruct {
 public:
  void __Delete__() {}

  ObjectMsgAllocator* mut_allocator() { return allocator_; }

 private:
  friend class ObjectMsgPtrUtil;
  void InitRefCount() { ref_cnt_ = 0; }
  void set_allocator(ObjectMsgAllocator* allocator) { allocator_ = allocator; }
  void IncreaseRefCount() { ref_cnt_++; }
  int32_t DecreaseRefCount() { return --ref_cnt_; }

  std::atomic<int32_t> ref_cnt_;
  ObjectMsgAllocator* allocator_;
};

struct ObjectMsgPtrUtil final {
  static void SetAllocator(ObjectMsgStruct* ptr, ObjectMsgAllocator* allocator) {
    ptr->set_allocator(allocator);
  }
  template<typename T>
  static void InitRef(T* ptr) {
    ptr->InitRefCount();
  }
  template<typename T>
  static void Ref(T* ptr) {
    ptr->IncreaseRefCount();
  }
  template<typename T>
  static void ReleaseRef(T* ptr) {
    if (ptr == nullptr) { return; }
    if (ptr->DecreaseRefCount() > 0) { return; }
    auto* allocator = ptr->mut_allocator();
    ptr->__Delete__();
    allocator->Deallocate(reinterpret_cast<char*>(ptr), sizeof(T));
  }
};

template<typename T>
struct ObjectMsgStructDefault final {
  ObjectMsgStructDefault() {
    std::memset(reinterpret_cast<void*>(Mutable()), 0, sizeof(T));
    ObjectMsgPtrUtil::InitRef<T>(Mutable());
  }

  const T& Get() const { return msg_; }
  T* Mutable() { return &msg_; }

 private:
  union {
    T msg_;
  };
};

template<bool is_object_msg>
struct ObjectMsgGetDefault final {
  template<typename T>
  static const T& Call(T* const* val) {
    return (*val)->__Default__();
  }
};
template<>
struct ObjectMsgGetDefault<false> final {
  template<typename T>
  static const T& Call(T const* val) {
    return *val;
  }
};

template<bool is_pointer>
struct ObjectMsgRecursiveNew {
  template<typename WalkCtxType, typename PtrFieldType>
  static void Call(WalkCtxType* ctx, PtrFieldType* field) {}
};

template<typename WalkCtxType, typename PtrFieldType>
struct _ObjectMsgRecursiveNew {
  static void Call(WalkCtxType* ctx, PtrFieldType* field, const char* field_name) {
    ObjectMsgRecursiveNew<
        std::is_pointer<PtrFieldType>::value
        && std::is_base_of<ObjectMsgStruct,
                           typename std::remove_pointer<PtrFieldType>::type>::value>::Call(ctx,
                                                                                           field);
  }
};

template<>
struct ObjectMsgRecursiveNew<true> {
  template<typename WalkCtxType, typename PtrFieldType>
  static void Call(WalkCtxType* ctx, PtrFieldType* field_ptr) {
    using FieldType = typename std::remove_pointer<PtrFieldType>::type;
    char* mem_ptr = ctx->Allocate(sizeof(FieldType));
    auto* ptr = new (mem_ptr) FieldType();
    *field_ptr = ptr;
    std::memset(reinterpret_cast<void*>(ptr), 0, sizeof(FieldType));
    ObjectMsgPtrUtil::InitRef<FieldType>(ptr);
    ObjectMsgPtrUtil::SetAllocator(ptr, ctx);
    ObjectMsgPtrUtil::Ref<FieldType>(ptr);
    ptr->template __WalkField__<_ObjectMsgRecursiveNew, WalkCtxType>(ctx);
  }
};

template<bool is_pointer>
struct ObjectMsgRecursiveRelease {
  template<typename WalkCtxType, typename PtrFieldType>
  static void Call(WalkCtxType* ctx, PtrFieldType* field) {}
};

template<typename WalkCtxType, typename PtrFieldType>
struct _ObjectMsgRecursiveRelease {
  static void Call(WalkCtxType* ctx, PtrFieldType* field, const char* field_name) {
    ObjectMsgRecursiveRelease<
        std::is_pointer<PtrFieldType>::value
        && std::is_base_of<ObjectMsgStruct,
                           typename std::remove_pointer<PtrFieldType>::type>::value>::Call(ctx,
                                                                                           field);
  }
};

template<>
struct ObjectMsgRecursiveRelease<true> {
  template<typename WalkCtxType, typename PtrFieldType>
  static void Call(WalkCtxType* ctx, PtrFieldType* field) {
    using FieldType = typename std::remove_pointer<PtrFieldType>::type;
    auto* ptr = *field;
    if (ptr == nullptr) { return; }
    ptr->template __ReverseWalkField__<_ObjectMsgRecursiveRelease, WalkCtxType>(ctx);
    ObjectMsgPtrUtil::ReleaseRef<FieldType>(ptr);
  }
};

template<typename T>
class ObjectMsgPtr final {
 public:
  ObjectMsgPtr() : ptr_(nullptr) {}
  ObjectMsgPtr(const ObjectMsgPtr& obj_ptr) {
    ptr_ = obj_ptr.ptr_;
    ObjectMsgPtrUtil::Ref<T>(ptr_);
  }
  ~ObjectMsgPtr() {
    ObjectMsgRecursiveRelease<true>::Call<ObjectMsgAllocator, T*>(ptr_->mut_allocator(), &ptr_);
  }

  static ObjectMsgPtr New() { return New(ObjectMsgDefaultAllocator::GlobalObjectMsgAllocator()); }

  static ObjectMsgPtr New(ObjectMsgAllocator* allocator) {
    ObjectMsgPtr ret;
    ObjectMsgRecursiveNew<true>::Call<ObjectMsgAllocator, T*>(allocator, &ret.ptr_);
    return ret;
  }

  ObjectMsgPtr& operator=(const ObjectMsgPtr& rhs) {
    ObjectMsgRecursiveRelease<true>::Call<ObjectMsgAllocator, T*>(ptr_->mut_allocator(), &ptr_);
    ptr_ = rhs.ptr_;
    ObjectMsgPtrUtil::Ref<T>(ptr_);
    return *this;
  }

  operator bool() const { return ptr_ == nullptr; }
  T* get() const { return ptr_; }
  T* operator->() const { return ptr_; }
  const T& operator*() const { return *ptr_; }

 private:
  T* ptr_;
};

template<typename T>
struct ObjectMsgIsScalar {
  const static bool value =
      std::is_arithmetic<T>::value || std::is_enum<T>::value || std::is_same<T, std::string>::value;
};
}

#endif  // ONEFLOW_CORE_COMMON_OBJECT_MSG_H_
