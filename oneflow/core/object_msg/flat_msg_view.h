/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_OBJECT_MSG_FLAT_MSG_VIEW_H_
#define ONEFLOW_CORE_OBJECT_MSG_FLAT_MSG_VIEW_H_

#include <vector>
#include <glog/logging.h>
#include "oneflow/core/object_msg/dss.h"
#include "oneflow/core/object_msg/flat_msg.h"
#include "oneflow/core/object_msg/struct_traits.h"
#include "oneflow/core/object_msg/static_counter.h"

namespace oneflow {

#define FLAT_MSG_VIEW_BEGIN(struct_name)                    \
  struct struct_name final {                                \
    using self_type = struct_name;                          \
    static const bool __is_flat_message_view_type__ = true; \
    FLAT_MSG_VIEW_DEFINE_BASIC_METHODS(struct_name);        \
    OF_PUBLIC DEFINE_STATIC_COUNTER(field_counter);         \
    DSS_BEGIN(STATIC_COUNTER(field_counter), struct_name);

#define FLAT_MSG_VIEW_END(struct_name)                                                    \
  static_assert(__is_flat_message_view_type__, "this struct is not a flat message view"); \
  OF_PUBLIC static const int __LastFieldIndex__ = STATIC_COUNTER(field_counter);          \
  OF_PUBLIC INCREASE_STATIC_COUNTER(field_counter);                                       \
  DSS_END(STATIC_COUNTER(field_counter), "flat message view", struct_name);               \
  }                                                                                       \
  ;

#define FLAT_MSG_VIEW_DEFINE_PATTERN(flat_msg_field_type, field_name)                        \
  static_assert(__is_flat_message_view_type__, "this struct is not a flat message view");    \
  _FLAT_MSG_VIEW_DEFINE_PATTERN(FLAT_MSG_TYPE_CHECK(flat_msg_field_type), field_name);       \
  OF_PUBLIC INCREASE_STATIC_COUNTER(field_counter);                                          \
  FLAT_MSG_VIEW_SPECIALIZE_FIELD_TYPE(STATIC_COUNTER(field_counter), flat_msg_field_type);   \
  FLAT_MSG_VIEW_CHECK_LAST_FIELD_TYPE(STATIC_COUNTER(field_counter), flat_msg_field_type);   \
  DSS_DEFINE_FIELD(STATIC_COUNTER(field_counter), "flat message view", flat_msg_field_type*, \
                   OF_PP_CAT(field_name, _));

#define FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(flat_msg_field_type, field_name)                  \
  FLAT_MSG_VIEW_DEFINE_VECTOR_PATTERN(flat_msg_field_type, field_name, FlatMsgViewPatternStdVec)

#define FLAT_MSG_VIEW_DEFINE_VECTOR_PATTERN(flat_msg_field_type, field_name, container_type)                  \
  static_assert(__is_flat_message_view_type__, "this struct is not a flat message view");       \
  _FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(FLAT_MSG_TYPE_CHECK(flat_msg_field_type), field_name, container_type); \
  OF_PUBLIC INCREASE_STATIC_COUNTER(field_counter);                                             \
  _SPECIALIZE_IS_REPEATED_PATTERN(STATIC_COUNTER(field_counter));                               \
  FLAT_MSG_VIEW_SPECIALIZE_FIELD_TYPE(STATIC_COUNTER(field_counter), flat_msg_field_type);      \
  FLAT_MSG_VIEW_CHECK_LAST_FIELD_TYPE(STATIC_COUNTER(field_counter), flat_msg_field_type);      \
  DSS_DEFINE_FIELD(STATIC_COUNTER(field_counter), "flat message view",                          \
                   container_type<flat_msg_field_type>, OF_PP_CAT(field_name, _));

// details

#define _FLAT_MSG_VIEW_DEFINE_PATTERN(field_type, field_name)                \
 public:                                                                     \
  const field_type& field_name() const { return *OF_PP_CAT(field_name, _); } \
                                                                             \
 private:                                                                    \
  const field_type* OF_PP_CAT(field_name, _);

#define _FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(field_type, field_name, container_type)                         \
 public:                                                                                       \
  const field_type& field_name(int i) const { return *OF_PP_CAT(field_name, _).at(i); }        \
  std::size_t OF_PP_CAT(field_name, _size)() const { return OF_PP_CAT(field_name, _).size(); } \
                                                                                               \
 private:                                                                                      \
  container_type<field_type> OF_PP_CAT(field_name, _);

#define FLAT_MSG_VIEW_DEFINE_BASIC_METHODS(T)        \
 public:                                             \
  template<int field_index, typename Enabled = void> \
  struct IsRepeatedPattern {                         \
    static const bool value = false;                 \
  };                                                 \
                                                     \
 private:                                            \
  template<int field_index, typename Enabled = void> \
  struct __FlatMsgViewFieldType__ {                  \
    struct type {};                                  \
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
using StdVector = std::vector<T>;

template<typename T, template<typename> class container_type>
struct FlatMsgViewPatternVec {
  using value_type = T;

  void __Init__() { new (&vec_buffer_) Vec(); }
  void __Delete__() { mut_vec()->~Vec(); }

  const T* at(int index) const { return vec().at(index); }
  size_t size() const { return vec().size(); }
  void clear() { mut_vec()->clear(); }
  void push_back(const T* ptr) { mut_vec()->push_back(ptr); }

 private:
  using Vec = container_type<const T*>;
  Vec* mut_vec() {
    Vec* __attribute__((__may_alias__)) ptr = reinterpret_cast<Vec*>(&vec_buffer_);
    return ptr;
  }

  const Vec& vec() const {
    const Vec* __attribute__((__may_alias__)) ptr = reinterpret_cast<const Vec*>(&vec_buffer_);
    return *ptr;
  }

  union {
    char vec_buffer_[sizeof(Vec)];
    int64_t align64_;
  };
};

template<typename T>
using FlatMsgViewPatternStdVec = FlatMsgViewPatternVec<T, StdVector>;

template<typename FlatMsgViewT, typename FlatMsgOneofField, typename OneofValueType>
class FlatMsgViewFieldCtx {
 public:
  using flat_msg_view_type = FlatMsgViewT;
  static_assert(std::is_same<OneofValueType, typename FlatMsgOneofField::struct_type>::value,
                "invalid view match");
  FlatMsgViewFieldCtx(const FlatMsgViewFieldCtx&) = delete;
  FlatMsgViewFieldCtx(FlatMsgViewFieldCtx&&) = delete;
  FlatMsgViewFieldCtx(const OneofValueType* repeated_flag_msg, std::size_t size)
      : repeated_flag_msg_(repeated_flag_msg), token_index_(0), size_(size) {}
  ~FlatMsgViewFieldCtx() = default;

  const OneofValueType* GetFlatMsg() const { return repeated_flag_msg_ + token_index_; }
  typename FlatMsgOneofField::field_type* GetOneof() const {
    return FlatMsgOneofField::FieldPtr4StructPtr(GetFlatMsg());
  }
  bool is_token_index_valid() const { return token_index_ < size_; }
  void increase_token_index() { ++token_index_; }
  int32_t token_count() const { return token_index_; }

 private:
  const OneofValueType* repeated_flag_msg_;
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
    using ConstFieldType = typename std::remove_pointer<FieldPtrT>::type;
    using FieldType = typename std::remove_const<ConstFieldType>::type;
    const auto* oneof = ctx->GetOneof();
    if (!oneof->template HasField<FieldType>()) { return true; }
    *field = &oneof->template GetField<FieldType>();
    ctx->increase_token_index();
    return false;
  }
};

template<typename WalkCtxType, typename FieldPtrT>
struct _FlatMsgViewFieldMatcher<true, WalkCtxType, FieldPtrT> {
  // return true if error occured
  static bool Call(WalkCtxType* ctx, FieldPtrT* field) {
    field->clear();
    using FieldType = typename FieldPtrT::value_type;
    while (ctx->is_token_index_valid()) {
      const auto* oneof = ctx->GetOneof();
      if (!oneof->template HasField<FieldType>()) { break; }
      field->push_back(&oneof->template GetField<FieldType>());
      ctx->increase_token_index();
    }
    return false;
  }
};

template<typename FlatMsgViewT, typename FlatMsgOneofField, typename ValueType>
struct FlatMsgViewUtil {
  static_assert(std::is_same<ValueType, typename FlatMsgOneofField::struct_type>::value,
                "invalid view match");
  static bool Match(FlatMsgViewT* flat_msg_view, const ValueType* data_ptr, std::size_t size) {
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
  static bool Match(FlatMsgViewT* self, const ContainerT& container) {
    return FlatMsgViewUtil<FlatMsgViewT, FlatMsgOneofField, typename ContainerT::value_type>::Match(
        self, container.data(), container.size());
  }
};

template<typename FlatMsgViewT, typename ValueType, typename Enabled>
struct FlatMsgViewContainerUtil<FlatMsgViewT, ValueType, StdVector<FlatMsg<ValueType>>, Enabled> {
  using FlatMsgOneofField =
      StructField<ValueType, typename ValueType::__OneofType, ValueType::__kDssFieldOffset>;
  static_assert(sizeof(ValueType) == sizeof(FlatMsg<ValueType>), "");
  static_assert(alignof(ValueType) == alignof(FlatMsg<ValueType>), "");
  static bool Match(FlatMsgViewT* self, const StdVector<FlatMsg<ValueType>>& container) {
    return FlatMsgViewUtil<FlatMsgViewT, FlatMsgOneofField, ValueType>::Match(
        self, &container.data()->Get(), container.size());
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
  FlatMsgView(const FlatMsgView&) = delete;
  FlatMsgView(FlatMsgView&&) = delete;
  static_assert(T::__is_flat_message_view_type__, "T is not a flat message view type");
  FlatMsgView() { view_.template __WalkField__<FlatMsgViewFieldInit>(&view_); }
  template<typename RepeatedFlatMsgT>
  explicit FlatMsgView(const RepeatedFlatMsgT& repeated_flat_msg) {
    view_.template __WalkField__<FlatMsgViewFieldInit>(&view_);
    CHECK(this->template Match(repeated_flat_msg));
  }
  ~FlatMsgView() { view_.template __ReverseWalkField__<FlatMsgViewFieldDelete>(&view_); }

  const T& operator*() const { return view_; }
  T& operator*() { return view_; }
  const T* operator->() const { return &view_; }
  T* operator->() { return &view_; }

  const T& Get() const { return view_; }
  T* Mutable() { return &view_; }

  template<typename RepeatedFlatMsgT>
  bool Match(const RepeatedFlatMsgT& repeated_flat_msg) {
    using OneofType = typename RepeatedFlatMsgT::value_type::self_value_type;
    return FlatMsgViewContainerUtil<T, OneofType, RepeatedFlatMsgT>::Match(&view_,
                                                                           repeated_flat_msg);
  }

 private:
  union {
    T view_;
  };
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_OBJECT_MSG_FLAT_MSG_VIEW_H_
