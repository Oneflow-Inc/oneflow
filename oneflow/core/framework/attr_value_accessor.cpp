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
#include "oneflow/core/framework/attr_value_accessor.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/stride.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/framework/attr_value.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {

namespace user_op {

// Basic and Enum Attr
#define BASIC_AND_ENUM_ATTR_SEQ_ENTRY(field, cpp_type, attr_type)                        \
  template<>                                                                             \
  cpp_type AttrValueAccessor<cpp_type>::Attr(const AttrValue& val) {                     \
    CHECK(val.has_##field());                                                            \
    return val.field();                                                                  \
  }                                                                                      \
  template<>                                                                             \
  void AttrValueAccessor<cpp_type>::Attr(const cpp_type& cpp_val, AttrValue* attr_val) { \
    attr_val->set_##field(cpp_val);                                                      \
  }

#define BASIC_AND_ENUM_ATTR_SEQ \
  BASIC_ATTR_SEQ                \
  ENUM_ATTR_SEQ

OF_PP_FOR_EACH_TUPLE(BASIC_AND_ENUM_ATTR_SEQ_ENTRY, BASIC_AND_ENUM_ATTR_SEQ)

#undef BASIC_AND_ENUM_ATTR_SEQ
#undef BASIC_AND_ENUM_ATTR_SEQ_ENTRY

// Customized Message Attr
template<>
Shape AttrValueAccessor<Shape>::Attr(const AttrValue& val) {
  return Shape(val.at_shape());
}

template<>
void AttrValueAccessor<Shape>::Attr(const Shape& cpp_val, AttrValue* attr_val) {
  cpp_val.ToProto(attr_val->mutable_at_shape());
}

template<>
Stride AttrValueAccessor<Stride>::Attr(const AttrValue& val) {
  return Stride(val.at_stride());
}

template<>
void AttrValueAccessor<Stride>::Attr(const Stride& cpp_val, AttrValue* attr_val) {
  cpp_val.ToProto(attr_val->mutable_at_stride());
}

template<>
Symbol<Device> AttrValueAccessor<Symbol<Device>>::Attr(const AttrValue& val) {
  auto pb_device = val.at_device();
  return CHECK_JUST(Device::New(*CHECK_JUST(DeviceTag4DeviceType(pb_device.device_type())),
                                pb_device.device_id()));
}

template<>
void AttrValueAccessor<Symbol<Device>>::Attr(const Symbol<Device>& cpp_val, AttrValue* attr_val) {
  attr_val->mutable_at_device()->set_device_type(cpp_val->enum_type());
  attr_val->mutable_at_device()->set_device_id(cpp_val->device_id());
}

// List of Basic Attr
#define LIST_BASIC_ATTR_SEQ_ENTRY(field, cpp_type, attr_type)                                   \
  template<>                                                                                    \
  cpp_type AttrValueAccessor<cpp_type>::Attr(const AttrValue& val) {                            \
    return PbRf2StdVec<cpp_type::value_type>(val.field().val());                                \
  }                                                                                             \
  template<>                                                                                    \
  void AttrValueAccessor<cpp_type>::Attr(const cpp_type& cpp_val, AttrValue* attr_val) {        \
    *(attr_val->mutable_##field()->mutable_val()) = StdVec2PbRf<cpp_type::value_type>(cpp_val); \
  }

OF_PP_FOR_EACH_TUPLE(LIST_BASIC_ATTR_SEQ_ENTRY, LIST_BASIC_ATTR_SEQ)

#undef LIST_BASIC_ATTR_SEQ_ENTRY

// List of Enum Attr
#define LIST_ENUM_ATTR_SEQ_ENTRY(field, cpp_type, attr_type)                                   \
  template<>                                                                                   \
  cpp_type AttrValueAccessor<cpp_type>::Attr(const AttrValue& val) {                           \
    std::vector<cpp_type::value_type> ret;                                                     \
    ret.reserve(val.field().val_size());                                                       \
    for (const auto& value : val.field().val()) {                                              \
      ret.emplace_back(static_cast<cpp_type::value_type>(value));                              \
    }                                                                                          \
    return ret;                                                                                \
  }                                                                                            \
  template<>                                                                                   \
  void AttrValueAccessor<cpp_type>::Attr(const cpp_type& cpp_val, AttrValue* attr_val) {       \
    using proto_type = std::remove_reference_t<decltype(attr_val->field().val())>::value_type; \
    std::vector<proto_type> vec;                                                               \
    vec.reserve(cpp_val.size());                                                               \
    for (const auto& value : cpp_val) { vec.emplace_back(static_cast<proto_type>(value)); }    \
    *(attr_val->mutable_##field()->mutable_val()) = StdVec2PbRf<proto_type>(vec);              \
  }

OF_PP_FOR_EACH_TUPLE(LIST_ENUM_ATTR_SEQ_ENTRY, LIST_ENUM_ATTR_SEQ)

#undef LIST_ENUM_ATTR_SEQ_ENTRY

// List of Customized Message Attr
template<>
std::vector<Shape> AttrValueAccessor<std::vector<Shape>>::Attr(const AttrValue& val) {
  std::vector<Shape> ret;
  ret.reserve(val.at_list_shape().val_size());
  for (const auto& value : val.at_list_shape().val()) { ret.emplace_back(value); }
  return ret;
}
template<>
void AttrValueAccessor<std::vector<Shape>>::Attr(const std::vector<Shape>& cpp_val,
                                                 AttrValue* attr_val) {
  attr_val->mutable_at_list_shape()->clear_val();
  FOR_RANGE(int32_t, i, 0, cpp_val.size()) {
    cpp_val.at(i).ToProto(attr_val->mutable_at_list_shape()->add_val());
  }
}
template<>
std::vector<Stride> AttrValueAccessor<std::vector<Stride>>::Attr(const AttrValue& val) {
  std::vector<Stride> ret;
  ret.reserve(val.at_list_stride().val_size());
  for (const auto& value : val.at_list_stride().val()) { ret.emplace_back(value); }
  return ret;
}
template<>
void AttrValueAccessor<std::vector<Stride>>::Attr(const std::vector<Stride>& cpp_val,
                                                  AttrValue* attr_val) {
  attr_val->mutable_at_list_stride()->clear_val();
  FOR_RANGE(int32_t, i, 0, cpp_val.size()) {
    cpp_val.at(i).ToProto(attr_val->mutable_at_list_stride()->add_val());
  }
}
// List of String Attr
template<>
std::vector<std::string> AttrValueAccessor<std::vector<std::string>>::Attr(const AttrValue& val) {
  return PbRpf2StdVec<std::string>(val.at_list_string().val());
}
template<>
void AttrValueAccessor<std::vector<std::string>>::Attr(const std::vector<std::string>& cpp_val,
                                                       AttrValue* attr_val) {
  *(attr_val->mutable_at_list_string()->mutable_val()) = StdVec2PbRpf<std::string>(cpp_val);
}
// ComplexDouble Attr
template<>
std::complex<double> AttrValueAccessor<std::complex<double>>::Attr(const AttrValue& val) {
  std::complex<double> ret{val.at_complex_double().real(), val.at_complex_double().imag()};
  return ret;
}
template<>
void AttrValueAccessor<std::complex<double>>::Attr(const std::complex<double>& cpp_val,
                                                   AttrValue* attr_val) {
  attr_val->mutable_at_complex_double()->set_real(cpp_val.real());
  attr_val->mutable_at_complex_double()->set_imag(cpp_val.imag());
}

template<typename ProtoT>
Maybe<AttrVal> MakeCppAttrValueFromProtoAttrValue(const ProtoT& attr_value) {
  switch (static_cast<int>(attr_value.value_case())) {
#define MAKE_ENTRY(field, T, attr_type)       \
  case static_cast<int>(attr_type):           \
    return std::static_pointer_cast<AttrVal>( \
        std::make_shared<TypedAttrVal<T>>(AttrValueAccessor<T>::Attr(attr_value)));
    OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, ATTR_SEQ);
#undef MAKE_ENTRY
    default: OF_UNIMPLEMENTED();
  }
}

/* static */ Maybe<AttrVal> AttrValueUtil::ToCppAttrValue(const AttrValue& proto_attr_value) {
  return MakeCppAttrValueFromProtoAttrValue(proto_attr_value);
}

/* static */ Maybe<void> AttrValueUtil::ToProtoAttrValue(const AttrVal& cpp_attr_value,
                                                         AttrValue* attr_value) {
  if (false) {
// clang-format off
#define MAKE_ENTRY(field, cpp_type, attr_type)                                        \
  }                                                                                   \
  else if (dynamic_cast<const TypedAttrValIf<cpp_type>*>(&cpp_attr_value) != nullptr) { \
    const auto* ptr = dynamic_cast<const TypedAttrValIf<cpp_type>*>(&cpp_attr_value);   \
    AttrValueAccessor<cpp_type>::Attr(ptr->val(), attr_value);
    OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, ATTR_SEQ);
#undef MAKE_ENTRY
    // clang-format on
  } else {
    OF_UNIMPLEMENTED();
  }
  return Maybe<void>::Ok();
}

}  // namespace user_op

}  // namespace oneflow
