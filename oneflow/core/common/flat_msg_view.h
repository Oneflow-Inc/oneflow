#ifndef ONEFLOW_CORE_COMMON_FLAT_MSG_VIEW_H_
#define ONEFLOW_CORE_COMMON_FLAT_MSG_VIEW_H_

#include "oneflow/core/common/dss.h"
#include "oneflow/core/common/flat_msg.h"
#include "oneflow/core/common/struct_traits.h"
#include "oneflow/core/common/static_counter.h"

namespace oneflow {

#define BEGIN_FLAT_MSG_VIEW(struct_name)                    \
  struct struct_name final {                                \
    using self_type = struct_name;                          \
    static const bool __is_flat_message_view_type__ = true; \
    FLAT_MSG_VIEW_DEFINE_BASIC_METHODS(struct_name);        \
    PRIVATE DEFINE_STATIC_COUNTER(field_counter);           \
    BEGIN_DSS(STATIC_COUNTER(field_counter), struct_name, 0);

#define END_FLAT_MSG_VIEW(struct_name)                                                    \
  static_assert(__is_flat_message_view_type__, "this struct is not a flat message view"); \
  PUBLIC static const int __NumberOfFields__ = STATIC_COUNTER(field_counter);             \
  PRIVATE INCREASE_STATIC_COUNTER(field_counter);                                         \
  END_DSS(STATIC_COUNTER(field_counter), "flat message view", struct_name);               \
  }                                                                                       \
  ;

#define FLAT_MSG_VIEW_DEFINE_PATTERN(flat_msg_field_type, field_name)                     \
  static_assert(__is_flat_message_view_type__, "this struct is not a flat message view"); \
  _FLAT_MSG_VIEW_DEFINE_PATTERN(FLAT_MSG_TYPE_CHECK(flat_msg_field_type), field_name);    \
  PRIVATE INCREASE_STATIC_COUNTER(field_counter);                                         \
  DSS_DEFINE_FIELD(STATIC_COUNTER(field_counter), "flat message view", OF_PP_CAT(field_name, _));

// details

#define _FLAT_MSG_VIEW_DEFINE_PATTERN(field_type, field_name)                        \
 public:                                                                             \
  const field_type& field_name() const { return *OF_PP_CAT(field_name, _); }         \
  field_type* OF_PP_CAT(mut_, field_name)() { return OF_PP_CAT(field_name, _); }     \
  field_type* OF_PP_CAT(mutable_, field_name)() { return OF_PP_CAT(field_name, _); } \
  template<typename T>                                                               \
  void OF_PP_CAT(set_, field_name)(T val) {                                          \
    static_assert(std::is_arithmetic<T>::value || std::is_enum<T>::value,            \
                  "only scalar fields have setter");                                 \
    *OF_PP_CAT(mut_, field_name)() = val;                                            \
  }                                                                                  \
                                                                                     \
 private:                                                                            \
  field_type* OF_PP_CAT(field_name, _);

#define FLAT_MSG_VIEW_DEFINE_BASIC_METHODS(T)                                                 \
 public:                                                                                      \
  void Clear() { std::memset(reinterpret_cast<void*>(this), 0, sizeof(T)); }                  \
  template<typename FlatMsgOneofField, typename RepeatedFlatMsgT>                             \
  bool MatchOneof(RepeatedFlatMsgT* repeated_flat_msg) {                                      \
    return FlatMsgViewUtil<self_type, FlatMsgOneofField, RepeatedFlatMsgT>::Match(            \
        this, repeated_flat_msg);                                                             \
  }                                                                                           \
  template<typename RepeatedFlatMsgT>                                                         \
  bool Match(RepeatedFlatMsgT* repeated_flat_msg) {                                           \
    using OneofField = StructField<typename RepeatedFlatMsgT::value_type,                     \
                                   typename RepeatedFlatMsgT::value_type::__OneofType,        \
                                   RepeatedFlatMsgT::value_type::__DssFieldOffset()>;         \
    return MatchOneof<OneofField>(repeated_flat_msg);                                         \
  }                                                                                           \
  template<typename FlatMsgOneofField, typename RepeatedFlatMsgT>                             \
  void InitOneof(RepeatedFlatMsgT* repeated_flat_msg) {                                       \
    FlatMsgViewUtil<self_type, FlatMsgOneofField, RepeatedFlatMsgT>::Init(this,               \
                                                                          repeated_flat_msg); \
  }                                                                                           \
  template<typename RepeatedFlatMsgT>                                                         \
  void Init(RepeatedFlatMsgT* repeated_flat_msg) {                                            \
    using OneofField = StructField<typename RepeatedFlatMsgT::value_type,                     \
                                   typename RepeatedFlatMsgT::value_type::__OneofType,        \
                                   RepeatedFlatMsgT::value_type::__DssFieldOffset()>;         \
    return InitOneof<OneofField>(repeated_flat_msg);                                          \
  }

template<typename FlatMsgOneofField, typename RepeatedFlatMsgT>
class FlatMsgViewFieldMatchCtx {
 public:
  static_assert(std::is_same<typename RepeatedFlatMsgT::value_type,
                             typename FlatMsgOneofField::struct_type>::value,
                "invalid view match");
  FlatMsgViewFieldMatchCtx(const FlatMsgViewFieldMatchCtx&) = delete;
  FlatMsgViewFieldMatchCtx(FlatMsgViewFieldMatchCtx&&) = delete;
  FlatMsgViewFieldMatchCtx(RepeatedFlatMsgT* repeated_flag_msg)
      : repeated_flag_msg_(repeated_flag_msg), field_index_(0) {}
  ~FlatMsgViewFieldMatchCtx() = default;

  typename RepeatedFlatMsgT::value_type* MutableFlatMsg() {
    return repeated_flag_msg_->Mutable(field_index_);
  }
  typename FlatMsgOneofField::field_type* MutableOneof() {
    return FlatMsgOneofField::FieldPtr4StructPtr(MutableFlatMsg());
  }
  void increase_field_index() { ++field_index_; }

 private:
  RepeatedFlatMsgT* repeated_flag_msg_;
  int field_index_;
};

template<int field_counter, typename WalkCtxType, typename FieldPtrT>
struct FlatMsgViewFieldMatcher {
  static bool Call(WalkCtxType* ctx, FieldPtrT* field) {
    using FieldType = typename std::remove_pointer<FieldPtrT>::type;
    auto* oneof = ctx->MutableOneof();
    if (!oneof->template HasField<FieldType>()) { return true; }
    *field = oneof->template MutableField<FieldType>();
    ctx->increase_field_index();
    return false;
  }
};

template<typename FlatMsgOneofField, typename RepeatedFlatMsgT>
class FlatMsgViewFieldInitCtx {
 public:
  static_assert(std::is_same<typename RepeatedFlatMsgT::value_type,
                             typename FlatMsgOneofField::struct_type>::value,
                "invalid view match");
  FlatMsgViewFieldInitCtx(const FlatMsgViewFieldInitCtx&) = delete;
  FlatMsgViewFieldInitCtx(FlatMsgViewFieldInitCtx&&) = delete;
  FlatMsgViewFieldInitCtx(RepeatedFlatMsgT* repeated_flag_msg)
      : repeated_flag_msg_(repeated_flag_msg) {}
  ~FlatMsgViewFieldInitCtx() = default;

  typename RepeatedFlatMsgT::value_type* AddFlatMsg() { return repeated_flag_msg_->Add(); }
  typename FlatMsgOneofField::field_type* AddOneof() {
    return FlatMsgOneofField::FieldPtr4StructPtr(AddFlatMsg());
  }

 private:
  RepeatedFlatMsgT* repeated_flag_msg_;
};

template<int field_counter, typename WalkCtxType, typename FieldPtrT>
struct FlatMsgViewFieldIniter {
  static void Call(WalkCtxType* ctx, FieldPtrT* field) {
    using FieldType = typename std::remove_pointer<FieldPtrT>::type;
    auto* oneof = ctx->AddOneof();
    *field = oneof->template MutableField<FieldType>();
  }
};

template<typename FlatMsgViewT, typename FlatMsgOneofField, typename RepeatedFlatMsgT>
struct FlatMsgViewUtil {
  static_assert(std::is_same<typename RepeatedFlatMsgT::value_type,
                             typename FlatMsgOneofField::struct_type>::value,
                "invalid view match");
  static_assert(FlatMsgViewT::__NumberOfFields__ <= RepeatedFlatMsgT::capacity, "invalid capacity");
  static bool Match(FlatMsgViewT* flat_msg_view, RepeatedFlatMsgT* repeated_flat_msg) {
    if (repeated_flat_msg->size() != FlatMsgViewT::__NumberOfFields__) { return false; }
    FlatMsgViewFieldMatchCtx<FlatMsgOneofField, RepeatedFlatMsgT> ctx(repeated_flat_msg);
    return !flat_msg_view->template __WalkFieldUntil__<FlatMsgViewFieldMatcher>(&ctx);
  }
  static void Init(FlatMsgViewT* flat_msg_view, RepeatedFlatMsgT* repeated_flat_msg) {
    CHECK(repeated_flat_msg->empty());
    FlatMsgViewFieldInitCtx<FlatMsgOneofField, RepeatedFlatMsgT> ctx(repeated_flat_msg);
    flat_msg_view->template __WalkField__<FlatMsgViewFieldIniter>(&ctx);
    CHECK_EQ(repeated_flat_msg->size(), FlatMsgViewT::__NumberOfFields__);
  }
};

template<typename T>
struct FlatMsgView final {
  static_assert(T::__is_flat_message_view_type__, "T is not a flat message view type");
  FlatMsgView() { view_.Clear(); }
  ~FlatMsgView() = default;
  template<typename RepeatedT>
  explicit FlatMsgView(RepeatedT* repeated_flag_msg) {
    view_.Clear();
    view_.Init(repeated_flag_msg);
  }

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
