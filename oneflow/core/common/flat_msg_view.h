#ifndef ONEFLOW_CORE_COMMON_FLAT_MSG_VIEW_H_
#define ONEFLOW_CORE_COMMON_FLAT_MSG_VIEW_H_

#include <vector>
#include "oneflow/core/common/dss.h"
#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/common/struct_traits.h"
#include "oneflow/core/common/static_counter.h"

namespace oneflow {

#define FLAT_MSG_VIEW_BEGIN(struct_name)                    \
  struct struct_name final {                                \
    using self_type = struct_name;                          \
    static const bool __is_flat_message_view_type__ = true; \
    FLAT_MSG_VIEW_DEFINE_BASIC_METHODS(struct_name);        \
    PRIVATE DEFINE_STATIC_COUNTER(field_counter);           \
    DSS_BEGIN(STATIC_COUNTER(field_counter), struct_name);

#define FLAT_MSG_VIEW_END(struct_name)                                                    \
  static_assert(__is_flat_message_view_type__, "this struct is not a flat message view"); \
  PUBLIC static const int __LastFieldIndex__ = STATIC_COUNTER(field_counter);             \
  PRIVATE INCREASE_STATIC_COUNTER(field_counter);                                         \
  DSS_END(STATIC_COUNTER(field_counter), "flat message view", struct_name);               \
  }                                                                                       \
  ;

#define FLAT_MSG_VIEW_DEFINE_PATTERN(flat_msg_field_type, field_name)                        \
  static_assert(__is_flat_message_view_type__, "this struct is not a flat message view");    \
  _FLAT_MSG_VIEW_DEFINE_PATTERN(FLAT_MSG_TYPE_CHECK(flat_msg_field_type), field_name);       \
  PRIVATE INCREASE_STATIC_COUNTER(field_counter);                                            \
  FLAT_MSG_VIEW_SPECIALIZE_FIELD_TYPE(STATIC_COUNTER(field_counter), flat_msg_field_type);   \
  FLAT_MSG_VIEW_CHECK_LAST_FIELD_TYPE(STATIC_COUNTER(field_counter), flat_msg_field_type);   \
  DSS_DEFINE_FIELD(STATIC_COUNTER(field_counter), "flat message view", flat_msg_field_type*, \
                   OF_PP_CAT(field_name, _));

#define FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(flat_msg_field_type, field_name)                  \
  static_assert(__is_flat_message_view_type__, "this struct is not a flat message view");       \
  _FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(FLAT_MSG_TYPE_CHECK(flat_msg_field_type), field_name); \
  PRIVATE INCREASE_STATIC_COUNTER(field_counter);                                               \
  _SPECIALIZE_IS_REPEATED_PATTERN(STATIC_COUNTER(field_counter));                               \
  FLAT_MSG_VIEW_SPECIALIZE_FIELD_TYPE(STATIC_COUNTER(field_counter), flat_msg_field_type);      \
  FLAT_MSG_VIEW_CHECK_LAST_FIELD_TYPE(STATIC_COUNTER(field_counter), flat_msg_field_type);      \
  DSS_DEFINE_FIELD(STATIC_COUNTER(field_counter), "flat message view",                          \
                   FlatMsgViewPatternVec<flat_msg_field_type>, OF_PP_CAT(field_name, _));

// details

#define _FLAT_MSG_VIEW_DEFINE_PATTERN(field_type, field_name)                \
 public:                                                                     \
  const field_type& field_name() const { return *OF_PP_CAT(field_name, _); } \
                                                                             \
 private:                                                                    \
  field_type* OF_PP_CAT(field_name, _);

#define _FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(field_type, field_name)                         \
 public:                                                                                       \
  const field_type& field_name(int i) const { return *OF_PP_CAT(field_name, _).at(i); }        \
  std::size_t OF_PP_CAT(field_name, _size)() const { return OF_PP_CAT(field_name, _).size(); } \
                                                                                               \
 private:                                                                                      \
  FlatMsgViewPatternVec<field_type> OF_PP_CAT(field_name, _);

#define FLAT_MSG_VIEW_DEFINE_BASIC_METHODS(T)                                               \
 public:                                                                                    \
  void Clear() { std::memset(reinterpret_cast<void*>(this), 0, sizeof(T)); }                \
  template<int field_index, typename Enabled = void>                                        \
  struct IsRepeatedPattern {                                                                \
    static const bool value = false;                                                        \
  };                                                                                        \
  template<typename RepeatedFlatMsgT>                                                       \
  bool Match(RepeatedFlatMsgT* repeated_flat_msg) {                                         \
    return FlatMsgViewContainerUtil<self_type,                                              \
                                    typename RepeatedFlatMsgT::value_type::self_value_type, \
                                    RepeatedFlatMsgT>::Match(this, repeated_flat_msg);      \
  }                                                                                         \
                                                                                            \
 private:                                                                                   \
  template<int field_index, typename Enabled = void>                                        \
  struct __FlatMsgViewFieldType__ {                                                         \
    struct type {};                                                                         \
  };

#define FLAT_MSG_VIEW_SPECIALIZE_FIELD_TYPE(field_index, field_type) \
 private:                                                            \
  template<typename Enabled>                                         \
  struct __FlatMsgViewFieldType__<field_index, Enabled> {            \
    using type = field_type;                                         \
  };

#define FLAT_MSG_VIEW_CHECK_LAST_FIELD_TYPE(field_index, field_type)                            \
 private:                                                                                       \
  static void OF_PP_CAT(__CheckLastFieldType__, __LINE__)() {                                   \
    static_assert(                                                                              \
        !(IsRepeatedPattern<field_index - 1>::value                                             \
          && std::is_same<__FlatMsgViewFieldType__<field_index - 1>::type, field_type>::value), \
        "repeated pattern shouldn't be followed by the pattern with same type");                \
  }

#define _SPECIALIZE_IS_REPEATED_PATTERN(field_index) \
  template<typename Enabled>                         \
  struct IsRepeatedPattern<field_index, Enabled> {   \
    static const bool value = true;                  \
  }

template<typename T>
struct FlatMsgViewPatternVec {
  using value_type = T;

  void __Init__() { new (&vec_buffer_) Vec(); }
  void __Delete__() { reinterpret_cast<Vec*>(&vec_buffer_)->~Vec(); }

  const T* at(int index) const { return reinterpret_cast<const Vec*>(&vec_buffer_)->at(index); }
  size_t size() const { return reinterpret_cast<const Vec*>(&vec_buffer_)->size(); }
  void push_back(const T* ptr) { reinterpret_cast<Vec*>(&vec_buffer_)->push_back(ptr); }

 private:
  using Vec = std::vector<const T*>;
  union {
    char vec_buffer_[sizeof(Vec)];
    int64_t align64_;
  };
};

template<typename FlatMsgViewT, typename FlatMsgOneofField, typename OneofValueType>
class FlatMsgViewFieldCtx {
 public:
  using flat_msg_view_type = FlatMsgViewT;
  static_assert(std::is_same<OneofValueType, typename FlatMsgOneofField::struct_type>::value,
                "invalid view match");
  FlatMsgViewFieldCtx(const FlatMsgViewFieldCtx&) = delete;
  FlatMsgViewFieldCtx(FlatMsgViewFieldCtx&&) = delete;
  FlatMsgViewFieldCtx(OneofValueType* repeated_flag_msg, std::size_t size)
      : repeated_flag_msg_(repeated_flag_msg), token_index_(0), size_(size) {}
  ~FlatMsgViewFieldCtx() = default;

  OneofValueType* MutableFlatMsg() { return repeated_flag_msg_ + token_index_; }
  typename FlatMsgOneofField::field_type* MutableOneof() {
    return FlatMsgOneofField::FieldPtr4StructPtr(MutableFlatMsg());
  }
  bool is_token_index_valid() const { return token_index_ < size_; }
  void increase_token_index() { ++token_index_; }
  int32_t token_count() const { return token_index_; }

 private:
  OneofValueType* repeated_flag_msg_;
  int32_t token_index_;
  const std::size_t size_;
};

template<bool is_repeated_pattern, typename WalkCtxType, typename FieldPtrT>
struct _FlatMsgViewFieldMatcher {};

template<int field_counter, typename WalkCtxType, typename FieldPtrT>
struct FlatMsgViewFieldMatcher {
  static const bool is_repeated_pattern =
      WalkCtxType::flat_msg_view_type::template IsRepeatedPattern<field_counter>::value;
  // return true if error occured
  static bool Call(WalkCtxType* ctx, FieldPtrT* field) {
    return _FlatMsgViewFieldMatcher<is_repeated_pattern, WalkCtxType, FieldPtrT>::Call(ctx, field);
  }
};

template<typename WalkCtxType, typename FieldPtrT>
struct _FlatMsgViewFieldMatcher<false, WalkCtxType, FieldPtrT> {
  // return true if error occured
  static bool Call(WalkCtxType* ctx, FieldPtrT* field) {
    if (!ctx->is_token_index_valid()) { return true; }
    using FieldType = typename std::remove_pointer<FieldPtrT>::type;
    auto* oneof = ctx->MutableOneof();
    if (!oneof->template HasField<FieldType>()) { return true; }
    *field = oneof->template MutableField<FieldType>();
    ctx->increase_token_index();
    return false;
  }
};

template<typename WalkCtxType, typename FieldPtrT>
struct _FlatMsgViewFieldMatcher<true, WalkCtxType, FieldPtrT> {
  // return true if error occured
  static bool Call(WalkCtxType* ctx, FieldPtrT* field) {
    using FieldType = typename FieldPtrT::value_type;
    while (ctx->is_token_index_valid()) {
      auto* oneof = ctx->MutableOneof();
      if (!oneof->template HasField<FieldType>()) { break; }
      field->push_back(oneof->template MutableField<FieldType>());
      ctx->increase_token_index();
    }
    return false;
  }
};

template<typename FlatMsgViewT, typename FlatMsgOneofField, typename ValueType>
struct FlatMsgViewUtil {
  static_assert(std::is_same<ValueType, typename FlatMsgOneofField::struct_type>::value,
                "invalid view match");
  static bool Match(FlatMsgViewT* flat_msg_view, ValueType* data_ptr, std::size_t size) {
    FlatMsgViewFieldCtx<FlatMsgViewT, FlatMsgOneofField, ValueType> ctx(data_ptr, size);
    bool ret = !flat_msg_view->template __WalkFieldUntil__<FlatMsgViewFieldMatcher>(&ctx);
    if (ret) {
      if (FlatMsgViewT::template IsRepeatedPattern<FlatMsgViewT::__LastFieldIndex__>::value) {
        ret = (ctx.token_count() == size)
              || /* last repeated field empty */ (ctx.token_count() - 1 == size);
      } else {
        ret = (ctx.token_count() == size);
      }
    }
    return ret;
  }
};

template<typename FlatMsgViewT, typename ValueType, typename ContainerT, typename Enabled = void>
struct FlatMsgViewContainerUtil {
  using FlatMsgOneofField =
      StructField<ValueType, typename ValueType::__OneofType, ValueType::__kDssFieldOffset>;
  static bool Match(FlatMsgViewT* self, ContainerT* container) {
    return FlatMsgViewUtil<FlatMsgViewT, FlatMsgOneofField, typename ContainerT::value_type>::Match(
        self, container->mut_data(), container->size());
  }
};

template<typename FlatMsgViewT, typename ValueType, typename Enabled>
struct FlatMsgViewContainerUtil<FlatMsgViewT, ValueType, std::vector<FlatMsg<ValueType>>, Enabled> {
  using FlatMsgOneofField =
      StructField<ValueType, typename ValueType::__OneofType, ValueType::__kDssFieldOffset>;
  static_assert(sizeof(ValueType) == sizeof(FlatMsg<ValueType>), "");
  static_assert(alignof(ValueType) == alignof(FlatMsg<ValueType>), "");
  static bool Match(FlatMsgViewT* self, std::vector<FlatMsg<ValueType>>* container) {
    return FlatMsgViewUtil<FlatMsgViewT, FlatMsgOneofField, ValueType>::Match(
        self, container->data()->Mutable(), container->size());
  }
};

template<bool is_repeated_pattern, typename FieldPtrT>
struct _FlatMsgViewFieldInit {};

template<int field_counter, typename WalkCtxType, typename FieldPtrT>
struct FlatMsgViewFieldInit {
  static const bool is_repeated_pattern =
      WalkCtxType::template IsRepeatedPattern<field_counter>::value;
  static void Call(WalkCtxType* ctx, FieldPtrT* field) {
    _FlatMsgViewFieldInit<is_repeated_pattern, FieldPtrT>::Call(field);
  }
};

template<typename FieldPtrT>
struct _FlatMsgViewFieldInit<false, FieldPtrT> {
  static void Call(FieldPtrT* field) {}
};

template<typename FieldPtrT>
struct _FlatMsgViewFieldInit<true, FieldPtrT> {
  static void Call(FieldPtrT* field) { field->__Init__(); }
};

template<bool is_repeated_pattern, typename FieldPtrT>
struct _FlatMsgViewFieldDelete {};

template<int field_counter, typename WalkCtxType, typename FieldPtrT>
struct FlatMsgViewFieldDelete {
  static const bool is_repeated_pattern =
      WalkCtxType::template IsRepeatedPattern<field_counter>::value;
  static void Call(WalkCtxType* ctx, FieldPtrT* field) {
    _FlatMsgViewFieldDelete<is_repeated_pattern, FieldPtrT>::Call(field);
  }
};

template<typename FieldPtrT>
struct _FlatMsgViewFieldDelete<false, FieldPtrT> {
  static void Call(FieldPtrT* field) {}
};

template<typename FieldPtrT>
struct _FlatMsgViewFieldDelete<true, FieldPtrT> {
  static void Call(FieldPtrT* field) { field->__Delete__(); }
};

template<typename T>
struct FlatMsgView final {
  static_assert(T::__is_flat_message_view_type__, "T is not a flat message view type");
  FlatMsgView() {
    view_.Clear();
    view_.template __WalkField__<FlatMsgViewFieldInit>(&view_);
  }
  ~FlatMsgView() { view_.template __ReverseWalkField__<FlatMsgViewFieldDelete>(&view_); }

  const T& operator*() const { return view_; }
  T& operator*() { return view_; }
  const T* operator->() const { return &view_; }
  T* operator->() { return &view_; }

  const T& Get() const { return view_; }
  T* Mutable() { return &view_; }

 private:
  union {
    T view_;
  };
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_FLAT_MSG_VIEW_H_
