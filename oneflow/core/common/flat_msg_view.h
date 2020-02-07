#ifndef ONEFLOW_CORE_COMMON_FLAT_MSG_VIEW_H_
#define ONEFLOW_CORE_COMMON_FLAT_MSG_VIEW_H_

#include "oneflow/core/common/dss.h"
#include "oneflow/core/common/flat_msg.h"

namespace oneflow {

#define BEGIN_FLAT_MSG_VIEW(struct_name) \
  struct FLAT_MSG_VIEW_TYPE(struct_name) final { \
    using self_type = FLAT_MSG_VIEW_TYPE(struct_name); \
    static const bool __is_flat_message_view_type__ = true;                 \
    FLAT_MSG_VIEW_DEFINE_BASIC_METHODS(FLAT_MSG_VIEW_TYPE(struct_name));\
    BEGIN_DSS(DSS_GET_FIELD_COUNTER(), FLAT_MSG_VIEW_TYPE(struct_name), 0); \

#define END_FLAT_MSG_VIEW(struct_name)                                               \
  static_assert(__is_flat_message_view_type__, "this struct is not a flat message view"); \
  END_DSS(DSS_GET_FIELD_COUNTER(), "flat message view", FLAT_MSG_VIEW_TYPE(struct_name)); \
  }                                                                             \
  ;

#define FLAT_MSG_VIEW_DEFINE_PATTERN(flat_msg_field_type, field_name) \
  static_assert(__is_flat_message_view_type__, "this struct is not a flat message view"); \
  _FLAT_MSG_VIEW_DEFINE_PATTERN(FLAT_MSG_TYPE(flat_msg_field_type), field_name);
  DSS_DEFINE_FIELD(DSS_GET_FIELD_COUNTER(), "flat message view", OF_PP_CAT(field_name, _));

// details

#define _FLAT_MSG_VIEW_DEFINE_PATTERN(field_type, field_name) \
 public: \
  const field_type& field_name() const { return *OF_PP_CAT(field_name, _); } \
  field_type* OF_PP_CAT(mut_, field_name)() { return OF_PP_CAT(field_name, _); } \
  field_type* OF_PP_CAT(mutable_, field_name)() { return OF_PP_CAT(field_name, _); } \
  void OF_PP_CAT(reset_, field_name)(field_type* val) { OF_PP_CAT(field_name, _) = val; } \
 private: \
  field_type* OF_PP_CAT(field_name, _);

#define FLAT_MSG_VIEW_DEFINE_BASIC_METHODS(T) \
 public:                                  \
  void Clear() { std::memset(reinterpret_cast<void*>(this), 0, sizeof(T)); } \
  template<typename RepeatedFlatMsgT, typename FlatMsgOneofField> \
  bool Match(RepeatedFlatMsgT* repeated_flat_msg) { \
    return FlatMsgViewUtil<RepeatedFlatMsgT, FlatMsgOneofField>::Match(this, repeated_flat_msg);\
  } \
  void Init(RepeatedFlatMsgT* repeated_flat_msg) { \
    FlatMsgViewUtil<RepeatedFlatMsgT, FlatMsgOneofField>::Init(this, repeated_flat_msg);\
  }

template<typename RepeatedFlatMsgT, typename FlatMsgOneofField>
class FlatMsgViewFieldMatchCtx {
 public:
  static_assert(std::is_same<typename RepeatedFlatMsgT::value_type, typename FlatMsgOneofField::struct_type>::value,
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
  int increase_field_index() { ++field_index_; }
  
 private:
  RepeatedFlatMsgT* repeated_flag_msg_;
  int field_index_;
};

template<int field_counter, typename WalkCtxType, typename FieldPtrT>
struct FlatMsgViewFieldMatcher {
  static bool Call(WalkCtxType* ctx, FieldPtrT* field, const char* field_name) {
    using FieldType = typename std::remove_pointer<FieldPtrT>::type;
    auto* oneof = ctx->MutableOneof();
    if (!oneof->HasField<FieldType>()) { return true; }
    *field = oneof->MutableField<FieldType>();
    ctx->increase_field_index();
    return false;
  }
};

template<typename FlatMsgViewT, typename RepeatedFlatMsgT, typename FlatMsgOneofField>
struct FlatMsgViewUtil {
  static_assert(std::is_same<typename RepeatedFlatMsgT::value_type, typename FlatMsgOneofField::struct_type>::value,
                "invalid view match");
  static_assert(FlatMsgViewT::__DSS__FieldCount() <= RepeatedFlatMsgT::capacity,
                "invalid capacity");
  static bool Match(FlatMsgViewT* flat_msg_view, RepeatedFlatMsgT* repeated_flat_msg) {
    if (repeated_flat_msg->size() != FlatMsgViewT::__DSS__FieldCount()) { return false; }
    FlatMsgViewFieldMatchCtx<RepeatedFlatMsgT, FlatMsgOneofField> ctx(repeated_flat_msg);
    flat_msg_view->__WalkFieldUntil__<FlatMsgViewFieldMatcher>(&ctx);
    return true;
  }
  static void Init(FlatMsgViewT* flat_msg_view, RepeatedFlatMsgT* repeated_flat_msg) {
    CHECK_EQ(repeated_flat_msg->size(), FlatMsgViewT::__DSS__FieldCount());
  }
};

}

#endif  // ONEFLOW_CORE_COMMON_FLAT_MSG_VIEW_H_
