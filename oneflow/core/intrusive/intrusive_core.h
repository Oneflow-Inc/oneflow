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
#ifndef ONEFLOW_CORE_INTRUSIVE_INTRUSIVE_CORE_H_
#define ONEFLOW_CORE_INTRUSIVE_INTRUSIVE_CORE_H_

#include <cstring>
#include <atomic>
#include <memory>
#include <type_traits>
#include <set>
#include <glog/logging.h>
#include "oneflow/core/intrusive/dss.h"
#include "oneflow/core/intrusive/static_counter.h"
#include "oneflow/core/intrusive/struct_traits.h"

namespace oneflow {

#define INTRUSIVE_BEGIN(class_name)                  \
  struct class_name final : public intrusive::Base { \
   public:                                           \
    using self_type = class_name;                    \
    static const bool __has_intrusive_ref__ = true;  \
    OF_PRIVATE DEFINE_STATIC_COUNTER(field_counter); \
    DSS_BEGIN(STATIC_COUNTER(field_counter), class_name);

#define INTRUSIVE_END(class_name)                                                   \
  _INTRUSIVE_DEFINE_REF();                                                          \
  static_assert(__has_intrusive_ref__, "this class is not intrusive-referenced");   \
  OF_PUBLIC static const int __NumberOfFields__ = STATIC_COUNTER(field_counter);    \
  OF_PRIVATE INCREASE_STATIC_COUNTER(field_counter);                                \
  DSS_END(STATIC_COUNTER(field_counter), "intrusive-referenced class", class_name); \
  }                                                                                 \
  ;

#define INTRUSIVE_DEFINE_FIELD(field_type, field_name)                                      \
 private:                                                                                   \
  static_assert(__has_intrusive_ref__, "this class is not intrusive-referenced");           \
  field_type field_name;                                                                    \
  INCREASE_STATIC_COUNTER(field_counter);                                                   \
  DSS_DEFINE_FIELD(STATIC_COUNTER(field_counter), "intrusive-referenced class", field_type, \
                   field_name);

#define INTRUSIVE_FIELD(struct_type, field_name)                             \
  StructField<struct_type, struct_type::OF_PP_CAT(field_name, DssFieldType), \
              struct_type::OF_PP_CAT(field_name, kDssFieldOffset)>

// Get field number by field name
// note: field numbers start from 1 instead of 0.
#define INTRUSIVE_FIELD_NUMBER(cls, field_name) cls::OF_PP_CAT(field_name, kDssFieldFieldNumber)

// Get field type by field number
#define INTRUSIVE_FIELD_TYPE(cls, field_number) cls::template __DssFieldType__<field_number>::type

// Get field offset by field number
#define INTRUSIVE_FIELD_OFFSET(cls, field_number) \
  cls::template __DssFieldOffset4FieldIndex__<field_number>::value

// Get current defined field counter inside a intrusive-referenced class.
// note: not used outside INTRUSIVE_BEGIN ... INTRUSIVE_END
// e.g.:
// INTRUSIVE_BEGIN(Foo);
//   static_assert(INTRUSIVE_FIELD_COUNTER == 0, "");
//   INTRUSIVE_DEFINE_FIELD(int64_t, a);
//   static_assert(INTRUSIVE_FIELD_COUNTER == 1, "");
//   INTRUSIVE_DEFINE_FIELD(int64_t, b);
//   static_assert(INTRUSIVE_FIELD_COUNTER == 2, "");
//   INTRUSIVE_DEFINE_FIELD(int8_t, c);
//   static_assert(INTRUSIVE_FIELD_COUNTER == 3, "");
//   INTRUSIVE_DEFINE_FIELD(int64_t, d);
// INTRUSIVE_END(Foo);
#define INTRUSIVE_FIELD_COUNTER STATIC_COUNTER(field_counter)

// details

#define _INTRUSIVE_DEFINE_REF()                                                                 \
 public:                                                                                        \
  intrusive::Ref* __mut_intrusive_ref__() { return &__intrusive_ref__; }                        \
  int32_t ref_cnt() const { return __intrusive_ref__.ref_cnt(); }                               \
                                                                                                \
 private:                                                                                       \
  intrusive::Ref __intrusive_ref__;                                                             \
  OF_PRIVATE INCREASE_STATIC_COUNTER(field_counter);                                            \
  DSS_DEFINE_FIELD(STATIC_COUNTER(field_counter), "intrusive-referenced class", intrusive::Ref, \
                   __intrusive_ref__);

#define _INTRUSIVE_DEFINE_FIELD(field_counter, field_type, field_name) \
 private:                                                              \
  field_type field_name;                                               \
  DSS_DEFINE_FIELD(field_counter, "intrusive-referenced class", field_type, field_name);

namespace intrusive {

struct Base {
  void __Init__() {}
  void __Delete__() {}
};

class Ref {
 public:
  int32_t ref_cnt() const { return ref_cnt_; }

 private:
  friend struct PtrUtil;
  void InitRefCount() { ref_cnt_ = 0; }
  void IncreaseRefCount() { ref_cnt_++; }
  int32_t DecreaseRefCount() { return --ref_cnt_; }

  std::atomic<int32_t> ref_cnt_;
};

struct PtrUtil final {
  template<typename T>
  static void NewAndInitRef(T** ptr) {
    *ptr = new T();
    (*ptr)->__mut_intrusive_ref__()->InitRefCount();
    Ref(*ptr);
  }
  template<typename T>
  static void Ref(T* ptr) {
    ptr->__mut_intrusive_ref__()->IncreaseRefCount();
  }
  template<typename T>
  static void ReleaseRef(T* ptr) {
    CHECK_NOTNULL(ptr);
    int32_t ref_cnt = ptr->__mut_intrusive_ref__()->DecreaseRefCount();
    if (ref_cnt > 0) { return; }
    ptr->__Delete__();
    delete ptr;
  }
};

}  // namespace intrusive

}  // namespace oneflow

#endif  // ONEFLOW_CORE_INTRUSIVE_INTRUSIVE_CORE_H_
