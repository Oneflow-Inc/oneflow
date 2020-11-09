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
#include "oneflow/core/framework/flex.h"
#include "oneflow/core/common/protobuf.h"
#include <iostream>

namespace oneflow {

std::shared_ptr<FlexDef> NewFlexDef(const FlexDefProto& flex_def_proto) {
  if (flex_def_proto.has_native_flex_def()) {
    auto flex_def = std::make_shared<NativeFlexDef>();
    flex_def->InitFromProto(flex_def_proto);
    return flex_def;
  } else if (flex_def_proto.has_struct_flex_def()) {
    auto flex_def = std::make_shared<StructFlexDef>();
    flex_def->InitFromProto(flex_def_proto);
    return flex_def;
  } else {
    UNIMPLEMENTED();
  }
  return std::shared_ptr<FlexDef>();
}

std::shared_ptr<FlexValue> FlexDef::New(FlexLabel label,
                                        const std::shared_ptr<const FlexDef>& flex_def) const {
  switch (label) {
    case kFlexLabelRequired:
    case kFlexLabelOptional: return New(flex_def);
    case kFlexLabelRepeated: return std::make_shared<RepeatedFlexValue>(flex_def);
    default: LOG(FATAL) << "UNIMPLEMENTED";
  }
  return std::shared_ptr<FlexValue>();
}

void NativeFlexDef::InitFromProto(const FlexDefProto& proto) {
  CHECK(proto.has_native_flex_def());
  native_flex_def_proto_ = proto.native_flex_def();
}

void NativeFlexDef::ToProto(FlexDefProto* proto) const {
  *proto->mutable_native_flex_def() = native_flex_def_proto_;
}

FlexCppType NativeFlexDef::cpp_type() const { return native_flex_def_proto_.cpp_type(); }

std::shared_ptr<FlexValue> NativeFlexDef::New(
    const std::shared_ptr<const FlexDef>& flex_def) const {
  CHECK_NOTNULL(dynamic_cast<const NativeFlexDef*>(flex_def.get()));
  return std::make_shared<NativeFlexValue>(flex_def);
}

void FlexFieldDef::InitFromProto(const FlexFieldDefProto& proto) {
  proto_ = proto;
  flex_def_ = NewFlexDef(proto.flex_def());
  auto value = flex_def_->New(proto.label(), flex_def_);
  value->InitFromProto(proto.default_val());
  default_val_ = value;
}

void FlexFieldDef::ToProto(FlexFieldDefProto* proto) const {
  *proto = proto_;
  flex_def_->ToProto(proto->mutable_flex_def());
  default_val_->ToProto(proto->mutable_default_val());
}

// Setters
FlexFieldDef* StructFlexDef::AddField(const std::string& field_name) {
  auto field = std::make_shared<FlexFieldDef>();
  fields_.push_back(field);
  CHECK(field_name2field_.emplace(field_name, field).second);
  return field.get();
}

std::shared_ptr<FlexValue> StructFlexDef::New(
    const std::shared_ptr<const FlexDef>& flex_def) const {
  CHECK_NOTNULL(dynamic_cast<const StructFlexDef*>(flex_def.get()));
  return std::make_shared<StructFlexValue>(flex_def);
}

// proto
void StructFlexDef::InitFromProto(const FlexDefProto& proto) {
  CHECK(proto.has_struct_flex_def());
  const auto& field_name2field = proto.struct_flex_def().field_name2field();
  for (const auto& pair : field_name2field) {
    auto field = std::make_shared<FlexFieldDef>();
    field->InitFromProto(pair.second);
    fields_.push_back(field);
    field_name2field_[pair.first] = field;
  }
}

void StructFlexDef::ToProto(FlexDefProto* proto) const {
  auto* struct_flex_def = proto->mutable_struct_flex_def();
  struct_flex_def->Clear();
  for (const auto& field : fields_) {
    field->ToProto(&(*struct_flex_def->mutable_field_name2field())[field->field_name()]);
  }
}

StructFlexDefBuilder& StructFlexDefBuilder::Field(
    FlexLabel label, const std::shared_ptr<const FlexDef>& flex_def, const std::string& field_name,
    const std::function<void(FlexValue*)>& SetDefaultVal, const std::string& description) {
  CHECK(static_cast<bool>(flex_def));
  auto* field = flex_def_->AddField(field_name);
  field->set_label(label);
  field->set_flex_def(flex_def);
  field->set_field_name(field_name);
  {
    auto default_val = flex_def->New(label, flex_def);
    SetDefaultVal(default_val.get());
    field->set_default_val(default_val);
  }
  field->set_description(description);
  return *this;
}

const RepeatedFlexValue& FlexValue::Repeated() const {
  const auto* ptr = reinterpret_cast<const RepeatedFlexValue*>(this);
  CHECK_NOTNULL(ptr);
  return *ptr;
}

RepeatedFlexValue* FlexValue::MutableRepeated() {
  auto* ptr = reinterpret_cast<RepeatedFlexValue*>(this);
  CHECK_NOTNULL(ptr);
  return ptr;
}

NativeFlexValue::NativeFlexValue(const std::shared_ptr<const FlexDef>& flex_def)
    : FlexValue(flex_def),
      value_case_(static_cast<NativeFlexValueProto::ValueCase>(
          std::dynamic_pointer_cast<const NativeFlexDef>(flex_def)->cpp_type())) {
  switch (value_case_) {
    case NativeFlexValueProto::kBoolVal: bool_val_ = false; return;
    case NativeFlexValueProto::kInt32Val: int32_val_ = 0; return;
    case NativeFlexValueProto::kInt64Val: int64_val_ = 0; return;
    case NativeFlexValueProto::kFloatVal: float_val_ = 0; return;
    case NativeFlexValueProto::kDoubleVal: double_val_ = 0; return;
    case NativeFlexValueProto::kDataTypeVal: data_type_val_ = kInvalidDataType; return;
    case NativeFlexValueProto::kStringVal: new (&string_val_buffer_) std::string(); return;
    case NativeFlexValueProto::kShapeVal: new (&shape_val_buffer_) Shape(); return;
    default: LOG(FATAL) << "UNIMPLEMENTED";
  }
}

NativeFlexValue::~NativeFlexValue() {
  switch (value_case_) {
    case NativeFlexValueProto::kBoolVal: bool_val_ = false; return;
    case NativeFlexValueProto::kInt32Val: int32_val_ = 0; return;
    case NativeFlexValueProto::kInt64Val: int64_val_ = 0; return;
    case NativeFlexValueProto::kFloatVal: float_val_ = 0; return;
    case NativeFlexValueProto::kDoubleVal: double_val_ = 0; return;
    case NativeFlexValueProto::kDataTypeVal: data_type_val_ = kInvalidDataType; return;
    case NativeFlexValueProto::kStringVal: {
      using String = std::string;
      String* ptr = reinterpret_cast<String*>(&string_val_buffer_);
      ptr->~String();
      return;
    }
    case NativeFlexValueProto::kShapeVal: {
      Shape* ptr = reinterpret_cast<Shape*>(&shape_val_buffer_);
      ptr->~Shape();
      return;
    }
    default: LOG(FATAL) << "UNIMPLEMENTED";
  }
}

bool NativeFlexValue::GetBool() const {
  CHECK_EQ(value_case_, NativeFlexValueProto::kBoolVal);
  return bool_val_;
}

int32_t NativeFlexValue::GetInt32() const {
  CHECK_EQ(value_case_, NativeFlexValueProto::kInt32Val);
  return int32_val_;
}

int64_t NativeFlexValue::GetInt64() const {
  CHECK_EQ(value_case_, NativeFlexValueProto::kInt64Val);
  return int64_val_;
}

float NativeFlexValue::GetFloat() const {
  CHECK_EQ(value_case_, NativeFlexValueProto::kFloatVal);
  return float_val_;
}

double NativeFlexValue::GetDouble() const {
  CHECK_EQ(value_case_, NativeFlexValueProto::kDoubleVal);
  return double_val_;
}

DataType NativeFlexValue::GetDataType() const {
  CHECK_EQ(value_case_, NativeFlexValueProto::kDataTypeVal);
  return data_type_val_;
}

const Shape& NativeFlexValue::GetShape() const {
  CHECK_EQ(value_case_, NativeFlexValueProto::kShapeVal);
  const Shape* ptr = reinterpret_cast<const Shape*>(&shape_val_buffer_[0]);
  return *ptr;
}

const std::string& NativeFlexValue::GetString() const {
  CHECK_EQ(value_case_, NativeFlexValueProto::kStringVal);
  const std::string* ptr = reinterpret_cast<const std::string*>(&shape_val_buffer_[0]);
  return *ptr;
}

void NativeFlexValue::SetBool(bool val) {
  CHECK_EQ(value_case_, NativeFlexValueProto::kBoolVal);
  bool_val_ = val;
}

void NativeFlexValue::SetInt32(int32_t val) {
  CHECK_EQ(value_case_, NativeFlexValueProto::kInt32Val);
  int32_val_ = val;
}

void NativeFlexValue::SetInt64(int64_t val) {
  CHECK_EQ(value_case_, NativeFlexValueProto::kInt64Val);
  int64_val_ = val;
}

void NativeFlexValue::SetFloat(float val) {
  CHECK_EQ(value_case_, NativeFlexValueProto::kFloatVal);
  float_val_ = val;
}

void NativeFlexValue::SetDouble(double val) {
  CHECK_EQ(value_case_, NativeFlexValueProto::kDoubleVal);
  double_val_ = val;
}

void NativeFlexValue::SetDataType(DataType val) {
  CHECK_EQ(value_case_, NativeFlexValueProto::kDataTypeVal);
  data_type_val_ = val;
}

void NativeFlexValue::SetShape(const Shape& val) {
  CHECK_EQ(value_case_, NativeFlexValueProto::kShapeVal);
  Shape* ptr = reinterpret_cast<Shape*>(&shape_val_buffer_[0]);
  *ptr = val;
}

void NativeFlexValue::SetString(const std::string& val) {
  CHECK_EQ(value_case_, NativeFlexValueProto::kStringVal);
  std::string* ptr = reinterpret_cast<std::string*>(&string_val_buffer_[0]);
  *ptr = val;
}

void NativeFlexValue::InitFromProto(const FlexValueProto& proto) {
  CHECK(proto.has_native_flex_value());
  const auto& native = proto.native_flex_value();
  CHECK_EQ(value_case_, native.value_case());
  switch (value_case_) {
    case NativeFlexValueProto::kBoolVal: return SetBool(native.bool_val());
    case NativeFlexValueProto::kInt32Val: return SetInt32(native.int32_val());
    case NativeFlexValueProto::kInt64Val: return SetInt64(native.int64_val());
    case NativeFlexValueProto::kFloatVal: return SetFloat(native.float_val());
    case NativeFlexValueProto::kDoubleVal: return SetDouble(native.double_val());
    case NativeFlexValueProto::kDataTypeVal: return SetDataType(native.data_type_val());
    case NativeFlexValueProto::kStringVal: return SetString(native.string_val());
    case NativeFlexValueProto::kShapeVal: return SetShape(Shape(native.shape_val()));
    default: LOG(FATAL) << "UNIMPLEMENTED";
  }
}

void NativeFlexValue::ToProto(FlexValueProto* proto) const {
  auto* native_flex_value = proto->mutable_native_flex_value();
  switch (value_case_) {
    case NativeFlexValueProto::kBoolVal: return native_flex_value->set_bool_val(GetBool());
    case NativeFlexValueProto::kInt32Val: return native_flex_value->set_int32_val(GetInt32());
    case NativeFlexValueProto::kInt64Val: return native_flex_value->set_int64_val(GetInt64());
    case NativeFlexValueProto::kFloatVal: return native_flex_value->set_float_val(GetFloat());
    case NativeFlexValueProto::kDoubleVal: return native_flex_value->set_double_val(GetDouble());
    case NativeFlexValueProto::kDataTypeVal:
      return native_flex_value->set_data_type_val(GetDataType());
    case NativeFlexValueProto::kStringVal: return native_flex_value->set_string_val(GetString());
    case NativeFlexValueProto::kShapeVal:
      return GetShape().ToProto(native_flex_value->mutable_shape_val());
    default: LOG(FATAL) << "UNIMPLEMENTED";
  }
}

const FlexValue& RepeatedFlexValue::Get(int64_t index) const { return *flex_values_.at(index); }

FlexValue* RepeatedFlexValue::Mutable(int64_t index) {
  FlexValue* ptr = flex_values_.at(index).get();
  return ptr;
}

FlexValue* RepeatedFlexValue::Add() {
  auto flex_value = flex_def()->New(flex_def());
  flex_values_.push_back(flex_value);
  return flex_value.get();
}

void RepeatedFlexValue::InitFromProto(const FlexValueProto& proto) {
  CHECK(proto.has_list_flex_value());
  for (const auto& flex_value : proto.list_flex_value().flex_value()) {
    Add()->InitFromProto(flex_value);
  }
}

void RepeatedFlexValue::ToProto(FlexValueProto* proto) const {
  auto* list_flex_value = proto->mutable_list_flex_value()->mutable_flex_value();
  for (const auto& flex_value : flex_values_) { flex_value->ToProto(list_flex_value->Add()); }
}

bool StructFlexValue::Defined(const std::string& field_name) const {
  return struct_flex_def().Has(field_name);
}

bool StructFlexValue::Has(const std::string& field_name) const {
  return field_name2flex_value_.find(field_name) != field_name2flex_value_.end();
}

const FlexValue& StructFlexValue::Get(const std::string& field_name) const {
  const auto& iter = field_name2flex_value_.find(field_name);
  if (iter == field_name2flex_value_.end()) {
    return *struct_flex_def().Field4FieldName(field_name).default_val();
  } else {
    return *iter->second;
  }
}

FlexValue* StructFlexValue::Mutable(const std::string& field_name) {
  const auto& iter = field_name2flex_value_.find(field_name);
  FlexValue* ptr = nullptr;
  if (iter == field_name2flex_value_.end()) {
    const auto& flex_field_def = struct_flex_def().Field4FieldName(field_name);
    auto field = flex_field_def.flex_def()->New(flex_field_def.label(), flex_field_def.flex_def());
    field_name2flex_value_[field_name] = field;
    ptr = field.get();
  } else {
    ptr = iter->second.get();
  }
  return ptr;
}

void StructFlexValue::InitFromProto(const FlexValueProto& proto) {
  for (const auto& pair : proto.struct_flex_value().field_name2flex_value()) {
    const auto& flex_field_def = struct_flex_def().Field4FieldName(pair.first);
    auto field = flex_field_def.flex_def()->New(flex_field_def.label(), flex_field_def.flex_def());
    field->InitFromProto(pair.second);
    CHECK(field_name2flex_value_.emplace(pair.first, field).second);
  }
}

void StructFlexValue::ToProto(FlexValueProto* proto) const {
  auto* map = proto->mutable_struct_flex_value()->mutable_field_name2flex_value();
  for (const auto& pair : field_name2flex_value_) { pair.second->ToProto(&(*map)[pair.first]); }
}

}  // namespace oneflow
