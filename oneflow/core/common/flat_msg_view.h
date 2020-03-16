#ifndef ONEFLOW_CORE_COMMON_FLAT_MSG_VIEW_H_
#define ONEFLOW_CORE_COMMON_FLAT_MSG_VIEW_H_

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
  PUBLIC static const int __NumberOfFields__ = STATIC_COUNTER(field_counter);             \
  PRIVATE INCREASE_STATIC_COUNTER(field_counter);                                         \
  DSS_END(STATIC_COUNTER(field_counter), "flat message view", struct_name);               \
  }                                                                                       \
  ;

#define FLAT_MSG_VIEW_DEFINE_PATTERN(flat_msg_field_type, field_name)                        \
  static_assert(__is_flat_message_view_type__, "this struct is not a flat message view");    \
  _FLAT_MSG_VIEW_DEFINE_PATTERN(FLAT_MSG_TYPE_CHECK(flat_msg_field_type), field_name);       \
  PRIVATE INCREASE_STATIC_COUNTER(field_counter);                                            \
  DSS_DEFINE_FIELD(STATIC_COUNTER(field_counter), "flat message view", flat_msg_field_type*, \
                   OF_PP_CAT(field_name, _));

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

#define FLAT_MSG_VIEW_DEFINE_BASIC_METHODS(T)                                                   \
 public:                                                                                        \
  void Clear() { std::memset(reinterpret_cast<void*>(this), 0, sizeof(T)); }                    \
  template<typename RepeatedFlatMsgT>                                                           \
  bool Match(RepeatedFlatMsgT* repeated_flat_msg) {                                             \
    return FlatMsgViewContainerUtil<self_type,                                                  \
                                    typename RepeatedFlatMsgT::value_type::self_value_type,     \
                                    RepeatedFlatMsgT>::Match(this, repeated_flat_msg);          \
  }                                                                                             \
  template<typename RepeatedFlatMsgT>                                                           \
  void Init(RepeatedFlatMsgT* repeated_flat_msg) {                                              \
    FlatMsgViewContainerUtil<self_type, typename RepeatedFlatMsgT::value_type::self_value_type, \
                             RepeatedFlatMsgT>::Init(this, repeated_flat_msg);                  \
  }

template<typename FlatMsgOneofField, typename OneofValueType>
class FlatMsgViewFieldCtx {
 public:
  static_assert(std::is_same<OneofValueType, typename FlatMsgOneofField::struct_type>::value,
                "invalid view match");
  FlatMsgViewFieldCtx(const FlatMsgViewFieldCtx&) = delete;
  FlatMsgViewFieldCtx(FlatMsgViewFieldCtx&&) = delete;
  FlatMsgViewFieldCtx(OneofValueType* repeated_flag_msg)
      : repeated_flag_msg_(repeated_flag_msg), field_index_(0) {}
  ~FlatMsgViewFieldCtx() = default;

  OneofValueType* MutableFlatMsg() { return repeated_flag_msg_ + field_index_; }
  typename FlatMsgOneofField::field_type* MutableOneof() {
    return FlatMsgOneofField::FieldPtr4StructPtr(MutableFlatMsg());
  }
  void increase_field_index() { ++field_index_; }
  int32_t field_count() const { return field_index_; }

 private:
  OneofValueType* repeated_flag_msg_;
  int32_t field_index_;
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

template<int field_counter, typename WalkCtxType, typename FieldPtrT>
struct FlatMsgViewFieldIniter {
  static void Call(WalkCtxType* ctx, FieldPtrT* field) {
    using FieldType = typename std::remove_pointer<FieldPtrT>::type;
    auto* oneof = ctx->MutableOneof();
    *field = oneof->template MutableField<FieldType>();
    ctx->increase_field_index();
  }
};

template<typename FlatMsgViewT, typename FlatMsgOneofField, typename ValueType>
struct FlatMsgViewUtil {
  static_assert(std::is_same<ValueType, typename FlatMsgOneofField::struct_type>::value,
                "invalid view match");
  static bool Match(FlatMsgViewT* flat_msg_view, ValueType* data_ptr, std::size_t size) {
    if (size != FlatMsgViewT::__NumberOfFields__) { return false; }
    FlatMsgViewFieldCtx<FlatMsgOneofField, ValueType> ctx(data_ptr);
    bool ret = !flat_msg_view->template __WalkFieldUntil__<FlatMsgViewFieldMatcher>(&ctx);
    if (ret) { CHECK_EQ(ctx.field_count(), FlatMsgViewT::__NumberOfFields__); }
    return ret;
  }
  static void Init(FlatMsgViewT* flat_msg_view, ValueType* data_ptr, std::size_t size) {
    CHECK_EQ(size, FlatMsgViewT::__NumberOfFields__);
    FlatMsgViewFieldCtx<FlatMsgOneofField, ValueType> ctx(data_ptr);
    flat_msg_view->template __WalkField__<FlatMsgViewFieldIniter>(&ctx);
    CHECK_EQ(ctx.field_count(), FlatMsgViewT::__NumberOfFields__);
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
  static void Init(FlatMsgViewT* self, ContainerT* container) {
    CHECK(container->empty());
    for (int i = 0; i < FlatMsgViewT::__NumberOfFields__; ++i) { container->Add(); }
    FlatMsgViewUtil<FlatMsgViewT, FlatMsgOneofField, typename ContainerT::value_type>::Init(
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
  static void Init(FlatMsgViewT* self, std::vector<FlatMsg<ValueType>>* container) {
    CHECK(container->empty());
    container->resize(FlatMsgViewT::__NumberOfFields__);
    FlatMsgViewUtil<FlatMsgViewT, FlatMsgOneofField, ValueType>::Init(
        self, container->data()->Mutable(), container->size());
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
