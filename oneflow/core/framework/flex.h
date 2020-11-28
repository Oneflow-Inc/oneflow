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
#ifndef ONEFLOW_CORE_FRAMEWORK_FLEX_H_
#define ONEFLOW_CORE_FRAMEWORK_FLEX_H_

#include <glog/logging.h>
#include <type_traits>
#include <memory>
#include <map>
#include "oneflow/core/framework/flex.pb.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

class Shape;
enum DataType;

class FlexDef;
class FlexValue;

class FlexDef {
 public:
  FlexDef(const FlexDef&) = delete;
  FlexDef(FlexDef&&) = delete;
  FlexDef() = default;
  virtual ~FlexDef() = default;

  virtual std::shared_ptr<FlexValue> New(const std::shared_ptr<const FlexDef>& flex_def) const = 0;
  virtual void InitFromProto(const FlexDefProto& proto) = 0;
  virtual void ToProto(FlexDefProto* proto) const = 0;
};
std::shared_ptr<FlexDef> NewFlexDef(const FlexDefProto& flex_def_proto);

class NativeFlexDef : public FlexDef {
 public:
  NativeFlexDef(const NativeFlexDef&) = delete;
  NativeFlexDef(NativeFlexDef&&) = delete;
  NativeFlexDef() = default;
  ~NativeFlexDef() override = default;

  NativeFlexType native_flex_type() const;

  void set_native_flex_type(NativeFlexType native_flex_type) {
    native_flex_def_proto_.set_native_flex_type(native_flex_type);
  }

  void InitFromProto(const FlexDefProto& proto) override;
  void ToProto(FlexDefProto* proto) const override;

  std::shared_ptr<FlexValue> New(const std::shared_ptr<const FlexDef>& flex_def) const override;

 private:
  NativeFlexDefProto native_flex_def_proto_;
};

class FlexFieldDef final {
 public:
  FlexFieldDef(const FlexFieldDef&) = delete;
  FlexFieldDef(FlexFieldDef&&) = delete;
  FlexFieldDef() = default;
  ~FlexFieldDef() = default;

  // Getters
  FlexLabel label() const { return proto_.label(); }
  const std::shared_ptr<const FlexDef>& flex_def() const { return flex_def_; }
  const std::string& field_name() const { return proto_.field_name(); }
  const std::shared_ptr<FlexValue>& default_val() const { return default_val_; }
  const std::string& description() const { return proto_.description(); }

  // Setters
  void set_label(FlexLabel label) { proto_.set_label(label); }
  void set_flex_def(const std::shared_ptr<const FlexDef>& flex_def) { flex_def_ = flex_def; }
  void set_field_name(const std::string& name) { proto_.set_field_name(name); }
  void set_default_val(const std::shared_ptr<FlexValue>& flex_value) { default_val_ = flex_value; }
  void set_description(const std::string& description) { proto_.set_description(description); }

  void InitFromProto(const FlexFieldDefProto& proto);

  void ToProto(FlexFieldDefProto* proto) const;

 private:
  FlexFieldDefProto proto_;
  std::shared_ptr<const FlexDef> flex_def_;
  std::shared_ptr<FlexValue> default_val_;
};

class StructFlexDef : public FlexDef {
 public:
  StructFlexDef(const StructFlexDef&) = delete;
  StructFlexDef(StructFlexDef&&) = delete;
  StructFlexDef() = default;
  ~StructFlexDef() override = default;

  bool Has(const std::string& field_name) const {
    return field_name2field_.find(field_name) != field_name2field_.end();
  }
  const std::vector<std::shared_ptr<const FlexFieldDef>>& fields() const { return fields_; }
  const FlexFieldDef& Field4FieldName(const std::string& field_name) const {
    return *field_name2field_.at(field_name);
  }

  // Setters
  FlexFieldDef* AddField(const std::string& field_name);

  // proto
  void InitFromProto(const FlexDefProto& proto);
  void ToProto(FlexDefProto* proto) const;

  std::shared_ptr<FlexValue> New(const std::shared_ptr<const FlexDef>& flex_def) const override;

 private:
  std::vector<std::shared_ptr<const FlexFieldDef>> fields_;
  std::map<std::string, std::shared_ptr<const FlexFieldDef>> field_name2field_;
};

class ListFlexDef : public FlexDef {
 public:
  ListFlexDef(const ListFlexDef&) = delete;
  ListFlexDef(ListFlexDef&&) = delete;
  ListFlexDef() = default;
  ListFlexDef(const std::shared_ptr<const FlexDef>& elem_flex_def)
      : FlexDef(), elem_flex_def_(elem_flex_def) {}
  ~ListFlexDef() override = default;

  const std::shared_ptr<const FlexDef> elem_flex_def() const { return elem_flex_def_; }

  // proto
  void InitFromProto(const FlexDefProto& proto);
  void ToProto(FlexDefProto* proto) const;

  std::shared_ptr<FlexValue> New(const std::shared_ptr<const FlexDef>& flex_def) const override;

 private:
  std::shared_ptr<const FlexDef> elem_flex_def_;
};

template<typename T, typename Enabled = void>
struct ScalarOrConstRef;

template<typename T>
struct ScalarOrConstRef<T, typename std::enable_if<std::is_scalar<T>::value>::type> final {
  using type = T;
};

template<typename T>
struct ScalarOrConstRef<T, typename std::enable_if<!std::is_scalar<T>::value>::type> final {
  using type = const T&;
};

template<typename T>
struct IsNativeFlexDef final {
  static const bool value = false;
};

template<typename T>
struct FlexDefBuilderTrait {
  static std::shared_ptr<const FlexDef> GetFlexDef() { return T::GetFlexDef(); }
};
template<typename T>
struct FlexValueSetter;

class StructFlexDefBuilder final {
 public:
  StructFlexDefBuilder(const StructFlexDefBuilder&) = delete;
  StructFlexDefBuilder(StructFlexDefBuilder&&) = default;
  StructFlexDefBuilder() : flex_def_(std::make_shared<StructFlexDef>()) {}
  explicit StructFlexDefBuilder(const std::shared_ptr<StructFlexDef>& flex_def)
      : flex_def_(flex_def) {}
  ~StructFlexDefBuilder() = default;

  StructFlexDefBuilder& Field(FlexLabel label, const std::shared_ptr<const FlexDef>& flex_def,
                              const std::string& field_name,
                              const std::function<void(FlexValue*)>& SetDefaultVal,
                              const std::string& description);

  template<typename T>
  StructFlexDefBuilder& Required(const std::string& field_name) {
    const auto& flex_def = FlexDefBuilderTrait<T>::GetFlexDef();
    return Required(flex_def, field_name);
  }
  StructFlexDefBuilder& Required(const std::shared_ptr<const FlexDef>& flex_def,
                                 const std::string& field_name) {
    return Field(kFlexLabelRequired, flex_def, field_name, [](FlexValue*) {}, "");
  }

  template<typename T>
  typename std::enable_if<IsNativeFlexDef<T>::value, StructFlexDefBuilder&>::type Optional(
      const std::string& field_name, typename ScalarOrConstRef<T>::type default_val) {
    const auto& flex_def = FlexDefBuilderTrait<T>::GetFlexDef();
    return Field(kFlexLabelOptional, flex_def, field_name,
                 [&](FlexValue* value) { FlexValueSetter<T>::Set(value, default_val); }, "");
  }
  template<typename T>
  typename std::enable_if<!IsNativeFlexDef<T>::value, StructFlexDefBuilder&>::type Optional(
      const std::string& field_name) {
    const auto& flex_def = T::GetFlexDef();
    return Optional(flex_def, field_name);
  }
  StructFlexDefBuilder& Optional(const std::shared_ptr<const FlexDef>& flex_def,
                                 const std::string& field_name) {
    return Field(kFlexLabelOptional, flex_def, field_name, [](FlexValue*) {}, "");
  }

  template<typename T>
  StructFlexDefBuilder& List(const std::string& field_name) {
    const auto& flex_def = FlexDefBuilderTrait<T>::GetFlexDef();
    return List(flex_def, field_name);
  }
  StructFlexDefBuilder& List(const std::shared_ptr<const FlexDef>& flex_def,
                             const std::string& field_name) {
    const auto list_flex_def = std::make_shared<ListFlexDef>(flex_def);
    return Field(kFlexLabelOptional, list_flex_def, field_name, [](FlexValue*) {}, "");
  }

  std::shared_ptr<const FlexDef> Build() { return flex_def_; }

 private:
  std::shared_ptr<StructFlexDef> flex_def_;
};

class FlexDefBuilder final {
 public:
  FlexDefBuilder() : flex_def_(std::make_shared<StructFlexDef>()) {}
  FlexDefBuilder(const std::shared_ptr<StructFlexDef>& flex_def) : flex_def_(flex_def) {}
  ~FlexDefBuilder() = default;

  StructFlexDefBuilder Struct() const { return StructFlexDefBuilder(flex_def_); }

 private:
  std::shared_ptr<StructFlexDef> flex_def_;
};

class ListFlexValue;
template<typename T>
struct FlexValueGetter;
class MutFlexValue;
class BaseMutListFlexValue;

class FlexValue {
 public:
  FlexValue(const FlexValue&) = delete;
  FlexValue(FlexValue&&) = delete;
  FlexValue(const std::shared_ptr<const FlexDef>& flex_def) : flex_def_(flex_def) {}
  virtual ~FlexValue() = default;

  const std::shared_ptr<const FlexDef>& flex_def() const { return flex_def_; }

  virtual bool Defined(const std::string& field_name) const { return false; }

  // for native flex value
  virtual bool GetBool() const = 0;
  virtual int64_t GetInt64() const = 0;
  virtual uint64_t GetUint64() const = 0;
  virtual double GetDouble() const = 0;
  virtual DataType GetDataType() const = 0;
  virtual const Shape& GetShape() const = 0;
  virtual const std::string& GetString() const = 0;
  virtual void SetBool(bool val) = 0;
  virtual void SetInt64(int64_t val) = 0;
  virtual void SetUint64(uint64_t val) = 0;
  virtual void SetDouble(double val) = 0;
  virtual void SetDataType(DataType val) = 0;
  virtual void SetShape(const Shape& val) = 0;
  virtual void SetString(const std::string& val) = 0;
  template<typename T>
  typename ScalarOrConstRef<T>::type Get() const {
    return FlexValueGetter<T>::Get(this);
  }

  // for list flex value
  virtual const FlexValue& Get(int64_t index) const = 0;
  virtual std::shared_ptr<FlexValue> Mutable(int64_t index) = 0;
  virtual std::shared_ptr<FlexValue> Add() = 0;
  template<typename T>
  typename ScalarOrConstRef<T>::type Get(int64_t index) const {
    return Get(index).Get<T>();
  }

  // for struct flex value
  virtual bool Has(const std::string& field_name) const = 0;
  virtual const FlexValue& Get(const std::string& field_name) const = 0;
  virtual const ListFlexValue& GetList(const std::string& field_name) const = 0;
  virtual std::shared_ptr<FlexValue> Mutable(const std::string& field_name) = 0;
  template<typename T>
  typename ScalarOrConstRef<T>::type Get(const std::string& field_name) const {
    return this->Get(field_name).Get<T>();
  }

  virtual void InitFromProto(const FlexValueProto& proto) = 0;
  virtual void ToProto(FlexValueProto* proto) const = 0;

 private:
  template<typename T>
  friend struct FlexValueSetter;
  friend class MutFlexValue;
  friend class BaseMutListFlexValue;

  template<typename T>
  void Set(const T& val) {
    return FlexValueSetter<T>::Set(this, val);
  }
  virtual std::shared_ptr<ListFlexValue> MutableList(const std::string& field_name) = 0;

  std::shared_ptr<const FlexDef> flex_def_;
};

class NativeFlexValue : public FlexValue {
 public:
  NativeFlexValue(const NativeFlexValue&) = delete;
  NativeFlexValue(NativeFlexValue&&) = delete;
  NativeFlexValue(const std::shared_ptr<const FlexDef>& flex_def);
  ~NativeFlexValue() override;

  // for native flex value
  bool GetBool() const override;
  int64_t GetInt64() const override;
  uint64_t GetUint64() const override;
  double GetDouble() const override;
  DataType GetDataType() const override;
  const Shape& GetShape() const override;
  const std::string& GetString() const override;
  template<typename T>
  typename ScalarOrConstRef<T>::type Get() const {
    return FlexValueGetter<T>::Get(this);
  }

  // for list flex value
  const FlexValue& Get(int64_t index) const override { LOG(FATAL) << "UNIMPLEMENTED"; }

  // for struct flex value
  bool Has(const std::string& field_name) const override { LOG(FATAL) << "UNIMPLEMENTED"; }
  const FlexValue& Get(const std::string& field_name) const override {
    LOG(FATAL) << "UNIMPLEMENTED";
  }
  const ListFlexValue& GetList(const std::string& field_name) const override {
    LOG(FATAL) << "UNIMPLEMENTED";
  }
  void ToProto(FlexValueProto* proto) const override;

 private:
  template<typename T>
  friend struct FlexValueSetter;
  friend class MutFlexValue;
  friend class BaseMutListFlexValue;
  void SetBool(bool val) override;
  void SetInt64(int64_t val) override;
  void SetUint64(uint64_t val) override;
  void SetDouble(double val) override;
  void SetDataType(DataType val) override;
  void SetShape(const Shape& val) override;
  void SetString(const std::string& val) override;
  const NativeFlexValueProto::ValueCase value_case_;

  std::shared_ptr<FlexValue> Mutable(int64_t index) override { LOG(FATAL) << "UNIMPLEMENTED"; }
  std::shared_ptr<FlexValue> Add() override { LOG(FATAL) << "UNIMPLEMENTED"; }
  std::shared_ptr<FlexValue> Mutable(const std::string& field_name) override {
    LOG(FATAL) << "UNIMPLEMENTED";
  }
  std::shared_ptr<ListFlexValue> MutableList(const std::string& field_name) override {
    LOG(FATAL) << "UNIMPLEMENTED";
  }
  template<typename T>
  void Set(const T& val) {
    return FlexValueSetter<T>::Set(this, val);
  }

  void InitFromProto(const FlexValueProto& proto) override;

  union {
    bool bool_val_;
    int32_t int32_val_;
    int64_t int64_val_;
    uint64_t uint64_val_;
    double double_val_;
    DataType data_type_val_;
    char string_val_buffer_[sizeof(std::string)];
    char shape_val_buffer_[sizeof(Shape)];
  };
};

class ListFlexValue : public FlexValue {
 public:
  ListFlexValue(const ListFlexValue&) = delete;
  ListFlexValue(ListFlexValue&&) = delete;
  ListFlexValue(const std::shared_ptr<const FlexDef>& flex_def)
      : FlexValue(flex_def),
        elem_flex_def_(std::dynamic_pointer_cast<const ListFlexDef>(flex_def)->elem_flex_def()) {}
  ~ListFlexValue() override = default;

  // for native flex value
  bool GetBool() const override { LOG(FATAL) << "UNIMPLEMENTED"; }
  int64_t GetInt64() const override { LOG(FATAL) << "UNIMPLEMENTED"; }
  uint64_t GetUint64() const override { LOG(FATAL) << "UNIMPLEMENTED"; }
  double GetDouble() const override { LOG(FATAL) << "UNIMPLEMENTED"; }
  DataType GetDataType() const override { LOG(FATAL) << "UNIMPLEMENTED"; }
  const Shape& GetShape() const override { LOG(FATAL) << "UNIMPLEMENTED"; }
  const std::string& GetString() const override { LOG(FATAL) << "UNIMPLEMENTED"; }

  // for list flex value
  const FlexValue& Get(int64_t index) const override;
  template<typename T>
  typename ScalarOrConstRef<T>::type Get(int64_t index) const {
    return Get(index).Get<T>();
  }
  using VecType = std::vector<std::shared_ptr<FlexValue>>;
  VecType::const_iterator begin() const { return flex_values_.begin(); }
  VecType::const_iterator end() const { return flex_values_.end(); }
  std::size_t size() const { return flex_values_.size(); }

  // for struct flex value
  bool Has(const std::string& field_name) const override { LOG(FATAL) << "UNIMPLEMENTED"; }
  const FlexValue& Get(const std::string& field_name) const override {
    LOG(FATAL) << "UNIMPLEMENTED";
  }
  const ListFlexValue& GetList(const std::string& field_name) const override {
    LOG(FATAL) << "UNIMPLEMENTED";
  }
  void ToProto(FlexValueProto* proto) const override;

  const std::shared_ptr<const FlexDef>& elem_flex_def() const { return elem_flex_def_; }

 private:
  template<typename T>
  friend struct FlexValueSetter;
  friend class MutFlexValue;
  friend class BaseMutListFlexValue;
  void SetBool(bool val) override { LOG(FATAL) << "UNIMPLEMENTED"; }
  void SetInt64(int64_t val) override { LOG(FATAL) << "UNIMPLEMENTED"; }
  void SetUint64(uint64_t val) override { LOG(FATAL) << "UNIMPLEMENTED"; }
  void SetDouble(double val) override { LOG(FATAL) << "UNIMPLEMENTED"; }
  void SetDataType(DataType val) override { LOG(FATAL) << "UNIMPLEMENTED"; }
  void SetShape(const Shape& val) override { LOG(FATAL) << "UNIMPLEMENTED"; }
  void SetString(const std::string& val) override { LOG(FATAL) << "UNIMPLEMENTED"; }
  std::shared_ptr<FlexValue> Mutable(int64_t index) override;
  std::shared_ptr<FlexValue> Add() override;
  std::shared_ptr<FlexValue> Mutable(const std::string& field_name) override {
    LOG(FATAL) << "UNIMPLEMENTED";
  }
  std::shared_ptr<ListFlexValue> MutableList(const std::string& field_name) override {
    LOG(FATAL) << "UNIMPLEMENTED";
  }
  void InitFromProto(const FlexValueProto& proto) override;

  VecType::iterator begin() { return flex_values_.begin(); }
  VecType::iterator end() { return flex_values_.end(); }

  const std::shared_ptr<const FlexDef> elem_flex_def_;
  std::vector<std::shared_ptr<FlexValue>> flex_values_;
};

class StructFlexValue : public FlexValue {
 public:
  StructFlexValue(const StructFlexValue&) = delete;
  StructFlexValue(StructFlexValue&&) = delete;
  StructFlexValue(const std::shared_ptr<const FlexDef>& flex_def)
      : FlexValue(flex_def), struct_flex_def_(dynamic_cast<const StructFlexDef*>(flex_def.get())) {
    CHECK_NOTNULL(struct_flex_def_);
  }
  ~StructFlexValue() override = default;

  // for native flex value
  bool GetBool() const override { LOG(FATAL) << "UNIMPLEMENTED"; }
  int64_t GetInt64() const override { LOG(FATAL) << "UNIMPLEMENTED"; }
  uint64_t GetUint64() const override { LOG(FATAL) << "UNIMPLEMENTED"; }
  double GetDouble() const override { LOG(FATAL) << "UNIMPLEMENTED"; }
  DataType GetDataType() const override { LOG(FATAL) << "UNIMPLEMENTED"; }
  const Shape& GetShape() const override { LOG(FATAL) << "UNIMPLEMENTED"; }
  const std::string& GetString() const override { LOG(FATAL) << "UNIMPLEMENTED"; }

  // for list flex value
  const FlexValue& Get(int64_t index) const override { LOG(FATAL) << "UNIMPLEMENTED"; }
  template<typename T>
  typename ScalarOrConstRef<T>::type Get(const std::string& field_name) const {
    return this->Get(field_name).Get<T>();
  }
  const ListFlexValue& GetList(const std::string& field_name) const {
    return dynamic_cast<const ListFlexValue&>(Get(field_name));
  }

  // for struct flex value
  bool Defined(const std::string& field_name) const override;
  bool Has(const std::string& field_name) const override;
  const FlexValue& Get(const std::string& field_name) const override;
  void ToProto(FlexValueProto* proto) const override;

 private:
  const StructFlexDef& struct_flex_def() const { return *struct_flex_def_; }

  void SetBool(bool val) override { LOG(FATAL) << "UNIMPLEMENTED"; }
  void SetInt64(int64_t val) override { LOG(FATAL) << "UNIMPLEMENTED"; }
  void SetUint64(uint64_t val) override { LOG(FATAL) << "UNIMPLEMENTED"; }
  void SetDouble(double val) override { LOG(FATAL) << "UNIMPLEMENTED"; }
  void SetDataType(DataType val) override { LOG(FATAL) << "UNIMPLEMENTED"; }
  void SetShape(const Shape& val) override { LOG(FATAL) << "UNIMPLEMENTED"; }
  void SetString(const std::string& val) override { LOG(FATAL) << "UNIMPLEMENTED"; }
  std::shared_ptr<FlexValue> Mutable(int64_t index) override { LOG(FATAL) << "UNIMPLEMENTED"; }
  std::shared_ptr<FlexValue> Add() override { LOG(FATAL) << "UNIMPLEMENTED"; }
  std::shared_ptr<FlexValue> Mutable(const std::string& field_name) override;
  std::shared_ptr<ListFlexValue> MutableList(const std::string& field_name) {
    return std::dynamic_pointer_cast<ListFlexValue>(Mutable(field_name));
  }

  void InitFromProto(const FlexValueProto& proto) override;

  const StructFlexDef* struct_flex_def_;
  std::map<std::string, std::shared_ptr<FlexValue>> field_name2flex_value_;
};

class BaseMutListFlexValue {
 public:
  BaseMutListFlexValue(const BaseMutListFlexValue&) = default;
  BaseMutListFlexValue(BaseMutListFlexValue&&) = default;
  BaseMutListFlexValue(const std::shared_ptr<ListFlexValue>& flex_value)
      : flex_value_(flex_value) {}
  ~BaseMutListFlexValue() = default;

  const FlexValue& Get(int64_t index) const { return flex_value_->Get(index); }
  std::shared_ptr<FlexValue> Mutable(int64_t index) { return flex_value_->Mutable(index); }
  std::shared_ptr<FlexValue> Add() { return flex_value_->Add(); }
  template<typename T>
  typename ScalarOrConstRef<T>::type Get(int64_t index) const {
    return flex_value_->Get<T>(index);
  }
  template<typename T>
  void Set(int64_t index, const T& val) {
    return flex_value_->Mutable(index)->Set<T>(val);
  }
  template<typename T>
  void Add(const T& val) {
    return flex_value_->Add()->Set<T>(val);
  }

  using VecType = std::vector<std::shared_ptr<FlexValue>>;
  VecType::const_iterator begin() const { return flex_value_->begin(); }
  VecType::const_iterator end() const { return flex_value_->end(); }
  std::size_t size() const { return flex_value_->size(); }

 private:
  std::shared_ptr<ListFlexValue> flex_value_;
};

class MutFlexValue final {
 public:
  MutFlexValue(const MutFlexValue&) = default;
  MutFlexValue(MutFlexValue&&) = default;
  MutFlexValue(const std::shared_ptr<FlexValue>& flex_value) : flex_value_(flex_value) {}
  ~MutFlexValue() = default;

  const std::shared_ptr<FlexValue>& Freeze() { return flex_value_; }

  // MutFlexValue can't be used in BaseMutListFlexValue, so MutList takes over Mutable Add
  class MutList final : public BaseMutListFlexValue {
   public:
    MutList(const MutList&) = default;
    MutList(MutList&&) = default;
    MutList(const std::shared_ptr<ListFlexValue>& flex_value) : BaseMutListFlexValue(flex_value) {}
    ~MutList() = default;

    MutFlexValue Mutable(int64_t index) { return BaseMutListFlexValue::Mutable(index); }
    MutFlexValue Add() { return BaseMutListFlexValue::Add(); }
    template<typename T>
    typename ScalarOrConstRef<T>::type Get(int64_t index) const {
      return BaseMutListFlexValue::Get<T>(index);
    }
    template<typename T>
    void Set(int64_t index, const T& val) {
      return BaseMutListFlexValue::Set<T>(index, val);
    }
    template<typename T>
    void Add(const T& val) {
      return BaseMutListFlexValue::Add<T>(val);
    }
  };

  const std::shared_ptr<const FlexDef>& flex_def() const { return flex_value_->flex_def(); }

  bool Defined(const std::string& field_name) const { return flex_value_->Defined(field_name); }

  // for native flex value
  template<typename T>
  typename ScalarOrConstRef<T>::type Get() const {
    return flex_value_->Get<T>();
  }
  template<typename T>
  void Set(const T& val) {
    return flex_value_->Set<T>(val);
  }

  // for list flex value
  const FlexValue& Get(int64_t index) const { return flex_value_->Get(index); }
  MutFlexValue Mutable(int64_t index) { return flex_value_->Mutable(index); }
  MutFlexValue Add() { return flex_value_->Add(); }
  template<typename T>
  typename ScalarOrConstRef<T>::type Get(int64_t index) const {
    return flex_value_->Get<T>(index);
  }
  template<typename T>
  void Set(int64_t index, const T& val) {
    return flex_value_->Mutable(index)->Set<T>(val);
  }
  template<typename T>
  void Add(const T& val) {
    return flex_value_->Add()->Set<T>(val);
  }

  // for struct flex value
  bool Has(const std::string& field_name) const { return flex_value_->Has(field_name); }
  const FlexValue& Get(const std::string& field_name) const { return flex_value_->Get(field_name); }
  const ListFlexValue& GetList(const std::string& field_name) const {
    return flex_value_->GetList(field_name);
  }
  MutFlexValue Mutable(const std::string& field_name) { return flex_value_->Mutable(field_name); }
  MutList MutableList(const std::string& field_name) {
    return flex_value_->MutableList(field_name);
  }
  template<typename T>
  typename ScalarOrConstRef<T>::type Get(const std::string& field_name) const {
    return flex_value_->Get<T>(field_name);
  }
  template<typename T>
  void Set(const std::string& field_name, const T& val) {
    return flex_value_->Mutable(field_name)->Set<T>(val);
  }

  void InitFromProto(const FlexValueProto& proto) { return flex_value_->InitFromProto(proto); }
  void ToProto(FlexValueProto* proto) const { return flex_value_->ToProto(proto); }

 private:
  std::shared_ptr<FlexValue> flex_value_;
};

using MutListFlexValue = MutFlexValue::MutList;

template<typename T>
MutFlexValue NewMutFlexValue() {
  return NewMutFlexValue(FlexDefBuilderTrait<T>::GetFlexDef());
}
inline MutFlexValue NewMutFlexValue(const std::shared_ptr<const FlexDef>& flex_def) {
  return flex_def->New(flex_def);
}

#define FLEX_DEF(customized_flex_struct, builder) \
  DECLARE_FLEX_DEF(customized_flex_struct);       \
  DEFINE_FLEX_DEF(customized_flex_struct, builder)

#define DECLARE_FLEX_DEF(customized_flex_struct)                                      \
  struct customized_flex_struct {                                                     \
    static MutFlexValue NewMutFlexValue() { return GetFlexDef()->New(GetFlexDef()); } \
    static std::shared_ptr<const FlexDef> GetFlexDef() {                              \
      static std::shared_ptr<StructFlexDef> flex_def;                                 \
      static std::atomic<bool> flex_def_allocated(false);                             \
      if (flex_def_allocated) { /* avoid circular reference deadlock */               \
        CHECK(static_cast<bool>(flex_def));                                           \
        return flex_def;                                                              \
      }                                                                               \
      static std::mutex mutex;                                                        \
      std::unique_lock<std::mutex> lock(mutex);                                       \
      flex_def.reset(new StructFlexDef());                                            \
      flex_def_allocated = true;                                                      \
      static FlexDefBuilder builder(flex_def);                                        \
      CHECK(MakeFlexDef(builder) == flex_def);                                        \
      return flex_def;                                                                \
    }                                                                                 \
    static std::shared_ptr<const FlexDef> MakeFlexDef(FlexDefBuilder& builder);       \
  }

#define DEFINE_FLEX_DEF(customized_flex_struct, builder) \
  std::shared_ptr<const FlexDef> customized_flex_struct::MakeFlexDef(FlexDefBuilder& builder)

#define SPECIALIZE_STRUCT_FLEX_DEF_UTIL(T, name)                                    \
  template<>                                                                        \
  struct IsNativeFlexDef<T> final {                                                 \
    static const bool value = true;                                                 \
  };                                                                                \
  template<>                                                                        \
  struct FlexDefBuilderTrait<T> {                                                   \
    static const std::shared_ptr<const FlexDef>& GetFlexDef() {                     \
      static const std::shared_ptr<const FlexDef> ptr = MakeFlexDef();              \
      return ptr;                                                                   \
    }                                                                               \
    static std::shared_ptr<const FlexDef> MakeFlexDef() {                           \
      auto flex_def = std::make_shared<NativeFlexDef>();                            \
      flex_def->set_native_flex_type(kNativeFlexType##name);                        \
      return flex_def;                                                              \
    }                                                                               \
  };                                                                                \
  template<>                                                                        \
  struct FlexValueGetter<T> final {                                                 \
    static typename ScalarOrConstRef<T>::type Get(const FlexValue* that) {          \
      return that->Get##name();                                                     \
    }                                                                               \
  };                                                                                \
  template<>                                                                        \
  struct FlexValueSetter<T> final {                                                 \
    static void Set(FlexValue* that, const T& val) { return that->Set##name(val); } \
  };

SPECIALIZE_STRUCT_FLEX_DEF_UTIL(bool, Bool);
SPECIALIZE_STRUCT_FLEX_DEF_UTIL(int8_t, Int64);
SPECIALIZE_STRUCT_FLEX_DEF_UTIL(int16_t, Int64);
SPECIALIZE_STRUCT_FLEX_DEF_UTIL(int32_t, Int64);
SPECIALIZE_STRUCT_FLEX_DEF_UTIL(int64_t, Int64);
SPECIALIZE_STRUCT_FLEX_DEF_UTIL(uint8_t, Uint64);
SPECIALIZE_STRUCT_FLEX_DEF_UTIL(uint16_t, Uint64);
SPECIALIZE_STRUCT_FLEX_DEF_UTIL(uint32_t, Uint64);
SPECIALIZE_STRUCT_FLEX_DEF_UTIL(uint64_t, Uint64);
SPECIALIZE_STRUCT_FLEX_DEF_UTIL(float, Double);
SPECIALIZE_STRUCT_FLEX_DEF_UTIL(double, Double);
SPECIALIZE_STRUCT_FLEX_DEF_UTIL(DataType, DataType);
SPECIALIZE_STRUCT_FLEX_DEF_UTIL(Shape, Shape);
SPECIALIZE_STRUCT_FLEX_DEF_UTIL(std::string, String);

#undef SPECIALIZE_STRUCT_FLEX_DEF_UTIL

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_FLEX_H_
