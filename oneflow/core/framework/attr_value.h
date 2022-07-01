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
#ifndef ONEFLOW_CORE_FRAMEWORK_ATTR_VALUE_H_
#define ONEFLOW_CORE_FRAMEWORK_ATTR_VALUE_H_

#include "oneflow/core/framework/user_op_attr.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/stride.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

template<typename T>
size_t HashTypedAttrVal(const T& val);

namespace user_op {

// SEQ
#define BASIC_ATTR_SEQ                                         \
  OF_PP_MAKE_TUPLE_SEQ(at_int32, int32_t, AttrType::kAtInt32)  \
  OF_PP_MAKE_TUPLE_SEQ(at_int64, int64_t, AttrType::kAtInt64)  \
  OF_PP_MAKE_TUPLE_SEQ(at_bool, bool, AttrType::kAtBool)       \
  OF_PP_MAKE_TUPLE_SEQ(at_float, float, AttrType::kAtFloat)    \
  OF_PP_MAKE_TUPLE_SEQ(at_double, double, AttrType::kAtDouble) \
  OF_PP_MAKE_TUPLE_SEQ(at_string, std::string, AttrType::kAtString)

#define ENUM_ATTR_SEQ OF_PP_MAKE_TUPLE_SEQ(at_data_type, DataType, AttrType::kAtDataType)

#define MESSAGE_ATTR_SEQ                                    \
  OF_PP_MAKE_TUPLE_SEQ(at_shape, Shape, AttrType::kAtShape) \
  OF_PP_MAKE_TUPLE_SEQ(at_stride, Stride, AttrType::kAtStride)

#define LIST_BASIC_ATTR_SEQ                                                         \
  OF_PP_MAKE_TUPLE_SEQ(at_list_int32, std::vector<int32_t>, AttrType::kAtListInt32) \
  OF_PP_MAKE_TUPLE_SEQ(at_list_int64, std::vector<int64_t>, AttrType::kAtListInt64) \
  OF_PP_MAKE_TUPLE_SEQ(at_list_float, std::vector<float>, AttrType::kAtListFloat)

#define LIST_ENUM_ATTR_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(at_list_data_type, std::vector<DataType>, AttrType::kAtListDataType)

#define LIST_MESSAGE_ATTR_SEQ                                                     \
  OF_PP_MAKE_TUPLE_SEQ(at_list_shape, std::vector<Shape>, AttrType::kAtListShape) \
  OF_PP_MAKE_TUPLE_SEQ(at_list_stride, std::vector<Stride>, AttrType::kAtListStride)

#define LIST_STRING_ATTR_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(at_list_string, std::vector<std::string>, AttrType::kAtListString)

#define ATTR_SEQ        \
  BASIC_ATTR_SEQ        \
  ENUM_ATTR_SEQ         \
  MESSAGE_ATTR_SEQ      \
  LIST_BASIC_ATTR_SEQ   \
  LIST_ENUM_ATTR_SEQ    \
  LIST_MESSAGE_ATTR_SEQ \
  LIST_STRING_ATTR_SEQ

// Type Trait: GetAttrType, GetCppType

template<typename T>
struct GetAttrType;

template<AttrType AttrT>
struct GetCppType;

#define SPECIALIZE_GET_ATTR_TYPE(field, type_cpp, type_proto)                     \
  template<>                                                                      \
  struct GetAttrType<type_cpp> : std::integral_constant<AttrType, type_proto> {}; \
  template<>                                                                      \
  struct GetCppType<type_proto> {                                                 \
    typedef type_cpp type;                                                        \
  };
OF_PP_FOR_EACH_TUPLE(SPECIALIZE_GET_ATTR_TYPE, ATTR_SEQ);
#undef SPECIALIZE_GET_ATTR_TYPE

class AttrVal {
 public:
  AttrVal() = default;
  virtual ~AttrVal() = default;

  virtual size_t hash_value() const = 0;
  virtual std::string ToString() const = 0;
  virtual bool operator==(const AttrVal& other) const = 0;
  bool operator!=(const AttrVal& other) const { return !(*this == other); }

 private:
  OF_DISALLOW_COPY_AND_MOVE(AttrVal);
};

template<typename T>
class TypedAttrValIf : public AttrVal {
 public:
  virtual const T& val() const = 0;
  size_t hash_value() const override { return std::hash<T>()(val()); }
  std::string ToString() const override {
    using ::operator<<;
    std::stringstream ss;
    ss << val();
    return ss.str();
  }

  bool operator==(const AttrVal& other) const override {
    auto* that = dynamic_cast<const TypedAttrValIf<T>*>(&other);
    if (that == nullptr) { return false; }
    return this->val() == that->val();
  }
};

template<typename T>
class TypedAttrVal final : public TypedAttrValIf<T> {
 public:
  TypedAttrVal(T v) : val_(v) {}
  ~TypedAttrVal() = default;

  const T& val() const override { return val_; }

 private:
  OF_DISALLOW_COPY_AND_MOVE(TypedAttrVal);

  T val_;
};

template<typename T>
class TypedAttrValRef final : public TypedAttrValIf<T> {
 public:
  TypedAttrValRef(const T* v) : val_(v) {}
  ~TypedAttrValRef() = default;

  const T& val() const override { return *val_; }

 private:
  OF_DISALLOW_COPY_AND_MOVE(TypedAttrValRef);

  const T* val_;
};

}  // namespace user_op

template<typename T>
const T& AttrValueCast(const user_op::AttrVal& val);

template<typename T>
std::shared_ptr<user_op::AttrVal> CastAttrValue(const T& attr_val);

template<typename T>
std::shared_ptr<user_op::AttrVal> CastAttrValue(const T* attr_val);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_ATTR_VALUE_H_
