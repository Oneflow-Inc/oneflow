#ifndef ONEFLOW_CORE_COMMON_OBJECT_MSG_CORE_H_
#define ONEFLOW_CORE_COMMON_OBJECT_MSG_CORE_H_

#include <cstring>
#include <atomic>
#include <memory>
#include <type_traits>
#include <glog/logging.h>
#include "oneflow/core/common/dss.h"

namespace oneflow {

#define BEGIN_OBJECT_MSG(class_name)                                                          \
  struct OBJECT_MSG_TYPE(class_name) final : public ObjectMsgStruct {                         \
   public:                                                                                    \
    static const bool __is_object_message_type__ = true;                                      \
    BEGIN_DSS(DSS_GET_FIELD_COUNTER(), OBJECT_MSG_TYPE(class_name), sizeof(ObjectMsgStruct)); \
    OBJECT_MSG_DEFINE_DEFAULT(OBJECT_MSG_TYPE(class_name));                                   \
    OBJECT_MSG_DEFINE_INIT();                                                                 \
    OBJECT_MSG_DEFINE_DELETE();

#define END_OBJECT_MSG(class_name)                                                  \
  static_assert(__is_object_message_type__, "this struct is not a object message"); \
  END_DSS(DSS_GET_FIELD_COUNTER(), "object message", OBJECT_MSG_TYPE(class_name));  \
  }                                                                                 \
  ;

#define OBJECT_MSG_DEFINE_OPTIONAL(field_type, field_name)                          \
  static_assert(__is_object_message_type__, "this struct is not a object message"); \
  OBJECT_MSG_DEFINE_ONEOF(OF_PP_CAT(field_name, __object_message_optional_field__), \
                          OBJECT_MSG_ONEOF_FIELD(field_type, field_name))

#define OBJECT_MSG_DEFINE_ONEOF(oneof_name, type_and_field_name_seq)                \
  static_assert(__is_object_message_type__, "this struct is not a object message"); \
  _OBJECT_MSG_DEFINE_ONEOF_FIELD(DSS_GET_FIELD_COUNTER(), oneof_name, type_and_field_name_seq)

#define OBJECT_MSG_ONEOF_FIELD(field_type, field_name) \
  OF_PP_MAKE_TUPLE_SEQ(OBJECT_MSG_TYPE(field_type), field_name)

#define OBJECT_MSG_PTR(class_name) ObjectMsgPtr<OBJECT_MSG_TYPE(class_name)>

#define OBJECT_MSG_TYPE(class_name) OF_PP_CAT(__object_message_type__, class_name)

// details

#define _OBJECT_MSG_DEFINE_ONEOF_FIELD(field_counter, oneof_name, type_and_field_name_seq) \
  OF_PP_FOR_EACH_TUPLE(OBJECT_MSG_TYPEDEF_ONEOF_FIELD, type_and_field_name_seq);           \
  OBJECT_MSG_DEFINE_ONEOF_ENUM_TYPE(oneof_name, type_and_field_name_seq);                  \
  OBJECT_MSG_DEFINE_ONEOF_UNION(oneof_name, type_and_field_name_seq);                      \
  OBJECT_MSG_DEFINE_ONEOF_ACCESSOR(field_counter, oneof_name, type_and_field_name_seq);    \
  OBJECT_MSG_DSS_DEFINE_UION_FIELD(field_counter, oneof_name, type_and_field_name_seq);

#define OBJECT_MSG_TYPEDEF_ONEOF_FIELD(field_type, field_name) \
  using OF_PP_CAT(field_name, _OneofFieldType) = typename OBJECT_MSG_STRUCT_MEMBER(field_type);

#define OBJECT_MSG_DSS_DEFINE_UION_FIELD(field_counter, oneof_name, type_and_field_name_seq) \
  DSS_DEFINE_FIELD(field_counter, "object message", OF_PP_CAT(oneof_name, _));               \
  DSS_DEFINE_UNION_FIELD_VISITOR(                                                            \
      field_counter, case_,                                                                  \
      OF_PP_FOR_EACH_TUPLE(OBJECT_MSG_MAKE_UNION_TYPE7FIELD4CASE, type_and_field_name_seq));

#define OBJECT_MSG_MAKE_UNION_TYPE7FIELD4CASE(field_type, field_name)                    \
  OF_PP_MAKE_TUPLE_SEQ(OF_PP_CAT(field_name, _OneofFieldType), OF_PP_CAT(field_name, _), \
                       _OBJECT_MSG_ONEOF_ENUM_VALUE(field_name))

#define OBJECT_MSG_DEFINE_ONEOF_ACCESSOR(field_counter, oneof_name, type_and_field_name_seq)   \
  _OBJECT_MSG_DEFINE_ONEOF_CASE_ACCESSOR(oneof_name, _OBJECT_MSG_ONEOF_ENUM_TYPE(oneof_name)); \
  OBJECT_MSG_MAKE_ONEOF_CLEARER(field_counter, oneof_name);                                    \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_OBJECT_MSG_ONEOF_ACCESSOR, (oneof_name),               \
                                   type_and_field_name_seq)

#define MAKE_OBJECT_MSG_ONEOF_ACCESSOR(oneof_name, pair) \
  OBJECT_MSG_MAKE_ONEOF_FIELD_GETTER(oneof_name, pair);  \
  OBJECT_MSG_MAKE_ONEOF_FIELD_CLEARER(oneof_name, pair); \
  OBJECT_MSG_MAKE_ONEOF_FIELD_MUTABLE(oneof_name, pair); \
  OBJECT_MSG_MAKE_ONEOF_FIELD_SETTER(oneof_name, pair);

#define OBJECT_MSG_MAKE_ONEOF_FIELD_SETTER(oneof_name, pair)    \
  template<typename T>                                          \
  void OF_PP_CAT(set_, OF_PP_PAIR_SECOND(pair))(const T& val) { \
    static_assert(!std::is_base_of<ObjectMsgStruct, T>::value,  \
                  "only setter of scalar field supported");     \
    *OF_PP_CAT(mutable_, OF_PP_PAIR_SECOND(pair))() = val;      \
  }

#define OBJECT_MSG_MAKE_ONEOF_FIELD_MUTABLE(oneof_name, pair)                                    \
 public:                                                                                         \
  OF_PP_PAIR_FIRST(pair) * OF_PP_CAT(mut_, OF_PP_PAIR_SECOND(pair))() {                          \
    auto* ptr = &OF_PP_CAT(oneof_name, _).OF_PP_CAT(OF_PP_PAIR_SECOND(pair), _);                 \
    using FieldType = typename std::remove_pointer<decltype(ptr)>::type;                         \
    static const bool is_ptr = std::is_pointer<FieldType>::value;                                \
    CHECK(OF_PP_CAT(has_, OF_PP_PAIR_SECOND(pair))());                                           \
    return MutableTrait<is_ptr>::Call(ptr);                                                      \
  }                                                                                              \
  OF_PP_PAIR_FIRST(pair) * OF_PP_CAT(mutable_, OF_PP_PAIR_SECOND(pair))() {                      \
    static const char* field_name = OF_PP_STRINGIZE(OF_PP_CAT(OF_PP_PAIR_SECOND(pair), _));      \
    auto* ptr = &OF_PP_CAT(oneof_name, _).OF_PP_CAT(OF_PP_PAIR_SECOND(pair), _);                 \
    using FieldType = typename std::remove_pointer<decltype(ptr)>::type;                         \
    static const bool is_ptr = std::is_pointer<FieldType>::value;                                \
    if (!OF_PP_CAT(has_, OF_PP_PAIR_SECOND(pair))()) {                                           \
      OF_PP_CAT(clear_, oneof_name)();                                                           \
      ObjectMsgNaiveInit<ObjectMsgAllocator, FieldType>::Call(mut_allocator(), ptr, field_name); \
      OF_PP_CAT(set_, OF_PP_CAT(oneof_name, _case))                                              \
      (_OBJECT_MSG_ONEOF_ENUM_VALUE(OF_PP_PAIR_SECOND(pair)));                                   \
    }                                                                                            \
    return MutableTrait<is_ptr>::Call(ptr);                                                      \
  }

#define OBJECT_MSG_MAKE_ONEOF_CLEARER(field_counter, oneof_name)                                \
 public:                                                                                        \
  void OF_PP_CAT(clear_, oneof_name)() {                                                        \
    const char* oneof_name_field = OF_PP_STRINGIZE(OF_PP_CAT(oneof_name, _));                   \
    __DSS__VisitField<field_counter, ObjectMsgField__Delete__, void,                            \
                      OF_PP_CAT(oneof_name, _UnionStructType)>::Call(nullptr,                   \
                                                                     &OF_PP_CAT(oneof_name, _), \
                                                                     oneof_name_field);         \
    OF_PP_CAT(set_, OF_PP_CAT(oneof_name, _case))                                               \
    (_OBJECT_MSG_ONEOF_NOT_SET_VALUE(oneof_name));                                              \
  }

#define OBJECT_MSG_MAKE_ONEOF_FIELD_CLEARER(oneof_name, pair)                            \
 public:                                                                                 \
  void OF_PP_CAT(clear_, OF_PP_PAIR_SECOND(pair))() {                                    \
    if (OF_PP_CAT(has_, OF_PP_PAIR_SECOND(pair))()) { OF_PP_CAT(clear_, oneof_name)(); } \
  }

#define OBJECT_MSG_MAKE_ONEOF_FIELD_GETTER(oneof_name, pair)                                      \
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
    return OF_PP_CAT(oneof_name, _case)()                                                         \
           == _OBJECT_MSG_ONEOF_ENUM_VALUE(OF_PP_PAIR_SECOND(pair));                              \
  }

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
  struct OF_PP_CAT(oneof_name, _UnionStructType) {                                     \
    union {                                                                            \
      OF_PP_FOR_EACH_TUPLE(MAKE_OBJECT_MSG_ONEOF_UNION_FIELD, type_and_field_name_seq) \
    };                                                                                 \
    _OBJECT_MSG_ONEOF_ENUM_TYPE(oneof_name) case_;                                     \
  } OF_PP_CAT(oneof_name, _);

#define MAKE_OBJECT_MSG_ONEOF_UNION_FIELD(field_type, field_name) \
  OBJECT_MSG_STRUCT_MEMBER(field_type) OF_PP_CAT(field_name, _);

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

#define OBJECT_MSG_STRUCT_MEMBER(class_name) \
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

#define OBJECT_MSG_DEFINE_DEFAULT(object_message_type_name)                           \
 public:                                                                              \
  const object_message_type_name& __Default__() const {                               \
    static const ObjectMsgStructDefault<object_message_type_name> default_object_msg; \
    return default_object_msg.Get();                                                  \
  }

#define OBJECT_MSG_DEFINE_INIT()                                            \
 public:                                                                    \
  template<typename WalkCtxType>                                            \
  void ObjectMsg__Init__(WalkCtxType* ctx) {                                \
    this->template __WalkField__<ObjectMsgField__Init__, WalkCtxType>(ctx); \
  }                                                                         \
                                                                            \
 private:                                                                   \
  template<int field_counter, typename WalkCtxType, typename PtrFieldType>  \
  struct ObjectMsgField__Init__ : public ObjectMsgNaiveInit<WalkCtxType, PtrFieldType> {};

#define OBJECT_MSG_OVERLOAD_INIT(field_counter, init_template)            \
 private:                                                                 \
  template<typename WalkCtxType, typename PtrFieldType>                   \
  struct ObjectMsgField__Init__<field_counter, WalkCtxType, PtrFieldType> \
      : public init_template<WalkCtxType, PtrFieldType> {};

#define OBJECT_MSG_DEFINE_DELETE()                                                \
 public:                                                                          \
  void ObjectMsg__Delete__() {                                                    \
    this->__Delete__();                                                           \
    this->template __ReverseWalkField__<ObjectMsgField__Delete__, void>(nullptr); \
  }                                                                               \
                                                                                  \
 private:                                                                         \
  template<int field_counter, typename WalkCtxType, typename PtrFieldType>        \
  struct ObjectMsgField__Delete__ : public ObjectMsgNaiveDelete<WalkCtxType, PtrFieldType> {};

#define OBJECT_MSG_OVERLOAD_DELETE(field_counter, delete_template)          \
 private:                                                                   \
  template<typename WalkCtxType, typename PtrFieldType>                     \
  struct ObjectMsgField__Delete__<field_counter, WalkCtxType, PtrFieldType> \
      : public delete_template<WalkCtxType, PtrFieldType> {};

class ObjectMsgAllocator {
 public:
  virtual ~ObjectMsgAllocator() {}
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

  int32_t ref_cnt() const { return ref_cnt_; }

 protected:
  ObjectMsgAllocator* mut_allocator() const { return allocator_; }

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
    CHECK_NOTNULL(ptr);
    int32_t ref_cnt = ptr->DecreaseRefCount();
    if (ref_cnt > 0) { return; }
    auto* allocator = ptr->mut_allocator();
    ptr->ObjectMsg__Delete__();
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
struct _ObjectMsgNaiveInit {
  template<typename WalkCtxType, typename PtrFieldType>
  static void Call(WalkCtxType* ctx, PtrFieldType* field) {}
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgNaiveInit {
  static void Call(WalkCtxType* ctx, PtrFieldType* field, const char* field_name) {
    static const bool is_ptr = std::is_pointer<PtrFieldType>::value;
    _ObjectMsgNaiveInit<is_ptr>::template Call<WalkCtxType, PtrFieldType>(ctx, field);
  }
};

template<>
struct _ObjectMsgNaiveInit<true> {
  template<typename WalkCtxType, typename PtrFieldType>
  static void Call(WalkCtxType* ctx, PtrFieldType* field_ptr) {
    static_assert(std::is_pointer<PtrFieldType>::value, "invalid use of _ObjectMsgNaiveInit");
    using FieldType = typename std::remove_pointer<PtrFieldType>::type;
    static_assert(std::is_base_of<ObjectMsgStruct, FieldType>::value,
                  "FieldType is not a subclass of ObjectMsgStruct");
    char* mem_ptr = ctx->Allocate(sizeof(FieldType));
    auto* ptr = new (mem_ptr) FieldType();
    *field_ptr = ptr;
    std::memset(reinterpret_cast<void*>(ptr), 0, sizeof(FieldType));
    ObjectMsgPtrUtil::InitRef<FieldType>(ptr);
    ObjectMsgPtrUtil::SetAllocator(ptr, ctx);
    ObjectMsgPtrUtil::Ref<FieldType>(ptr);
    ptr->template ObjectMsg__Init__<WalkCtxType>(ctx);
  }
};

template<bool is_pointer>
struct _ObjectMsgNaiveDelete {
  template<typename WalkCtxType, typename PtrFieldType>
  static void Call(WalkCtxType* ctx, PtrFieldType* field) {}
};

template<typename WalkCtxType, typename PtrFieldType>
struct ObjectMsgNaiveDelete {
  static void Call(WalkCtxType* ctx, PtrFieldType* field, const char* field_name) {
    static const bool is_ptr = std::is_pointer<PtrFieldType>::value;
    _ObjectMsgNaiveDelete<is_ptr>::template Call<WalkCtxType, PtrFieldType>(ctx, field);
  }
};

template<>
struct _ObjectMsgNaiveDelete<true> {
  template<typename WalkCtxType, typename PtrFieldType>
  static void Call(WalkCtxType* ctx, PtrFieldType* field) {
    static_assert(std::is_pointer<PtrFieldType>::value, "invalid use of _ObjectMsgNaiveDelete");
    using FieldType = typename std::remove_pointer<PtrFieldType>::type;
    static_assert(std::is_base_of<ObjectMsgStruct, FieldType>::value,
                  "FieldType is not a subclass of ObjectMsgStruct");
    auto* ptr = *field;
    if (ptr == nullptr) { return; }
    ObjectMsgPtrUtil::ReleaseRef<FieldType>(ptr);
  }
};

template<typename T>
class ObjectMsgPtr final {
 public:
  ObjectMsgPtr() : ptr_(nullptr) {}
  ObjectMsgPtr(T* ptr) : ptr_(nullptr) { Reset(ptr); }
  ObjectMsgPtr(const ObjectMsgPtr& obj_ptr) {
    ptr_ = nullptr;
    Reset(obj_ptr.ptr_);
  }
  ObjectMsgPtr(ObjectMsgPtr&& obj_ptr) {
    ptr_ = obj_ptr.ptr_;
    obj_ptr.ptr_ = nullptr;
  }
  ~ObjectMsgPtr() { Clear(); }

  operator bool() const { return ptr_ != nullptr; }
  const T* Get() const { return ptr_; }
  const T* operator->() const { return ptr_; }
  const T& operator*() const { return *ptr_; }
  bool operator==(const ObjectMsgPtr& rhs) const { return this->ptr_ == rhs.ptr_; }

  T* Mutable() { return ptr_; }
  T* operator->() { return ptr_; }
  T& operator*() { return *ptr_; }

  void Reset() { Reset(nullptr); }

  void Reset(T* ptr) {
    Clear();
    if (ptr == nullptr) { return; }
    ptr_ = ptr;
    ObjectMsgPtrUtil::Ref<T>(ptr_);
  }

  ObjectMsgPtr& operator=(const ObjectMsgPtr& rhs) {
    Reset(rhs.ptr_);
    return *this;
  }

  static ObjectMsgPtr New() { return New(ObjectMsgDefaultAllocator::GlobalObjectMsgAllocator()); }
  static ObjectMsgPtr New(ObjectMsgAllocator* allocator) {
    ObjectMsgPtr ret;
    ObjectMsgNaiveInit<ObjectMsgAllocator, T*>::Call(allocator, &ret.ptr_, nullptr);
    return ret;
  }

 private:
  void Clear() {
    if (ptr_ == nullptr) { return; }
    ObjectMsgPtrUtil::ReleaseRef<T>(ptr_);
    ptr_ = nullptr;
  }
  T* ptr_;
};

template<typename T>
struct ObjectMsgIsScalar {
  const static bool value =
      std::is_arithmetic<T>::value || std::is_enum<T>::value || std::is_same<T, std::string>::value;
};
}

#endif  // ONEFLOW_CORE_COMMON_OBJECT_MSG_CORE_H_
