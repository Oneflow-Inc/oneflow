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
#ifndef ONEFLOW_XRT_ANY_H_
#define ONEFLOW_XRT_ANY_H_

#include <functional>
#include <type_traits>
#include <typeinfo>

#include "glog/logging.h"

namespace oneflow {
namespace xrt {

class Any {
 public:
  inline Any() = default;

  inline Any(Any&& other);

  inline Any(const Any& other);

  template<typename T>
  inline Any(T&& value);

  inline virtual ~Any();

  inline Any& operator=(Any&& other);

  inline Any& operator=(const Any& other);

  template<typename T>
  inline Any& operator=(T&& value);

  inline void Swap(Any& other);

  template<typename T>
  inline const T& Cast() const;

  template<typename T>
  inline T& Cast();

  template<typename T>
  inline friend const T& any_cast(const Any& any);
  template<typename T>
  inline friend T& any_cast(Any& any);

 private:
  struct AnyType {
    const std::type_info* ptype_info;
  };

  struct AnyData {
    virtual ~AnyData() = default;
    virtual const void* Ptr() { return nullptr; };
    std::function<AnyData*()> clone;
  };

  template<typename T>
  struct AnyDataImpl : public AnyData {
    T data_content;
    explicit AnyDataImpl(const T& value);
    const void* Ptr() override { return &data_content; }
  };

  template<typename T>
  inline AnyType TypeInfo() const;

  template<typename T>
  inline bool CheckType() const;

 private:
  AnyType type_;
  AnyData* data_ = nullptr;
};

template<typename T>
Any::AnyDataImpl<T>::AnyDataImpl(const T& value) : data_content(value) {
  this->clone = [this]() -> Any::AnyDataImpl<T>* { return new AnyDataImpl<T>(this->data_content); };
}

void Any::Swap(Any& other) {
  std::swap(type_, other.type_);
  std::swap(data_, other.data_);
}

Any::Any(Any&& other) { this->Swap(other); }

Any::Any(const Any& other) {
  type_ = other.type_;
  if (other.data_) { data_ = other.data_->clone(); }
}

Any::~Any() {
  if (data_) delete data_;
  data_ = nullptr;
}

template<typename T>
Any::AnyType Any::TypeInfo() const {
  Any::AnyType type;
  type.ptype_info = &typeid(T);
  return std::move(type);
}

template<typename T>
Any::Any(T&& value) {
  typedef typename std::decay<T>::type DT;
  if (std::is_same<DT, Any>::value) {
    *this = std::move(value);
  } else {
    type_ = TypeInfo<T>();
    data_ = new AnyDataImpl<T>(value);
  }
}

Any& Any::operator=(Any&& other) {
  Any(std::move(other)).Swap(*this);
  return *this;
}

Any& Any::operator=(const Any& other) {
  Any(other).Swap(*this);
  return *this;
}

template<typename T>
Any& Any::operator=(T&& value) {
  Any(std::move(value)).Swap(*this);
  return *this;
}

template<typename T>
bool Any::CheckType() const {
  if (typeid(T).hash_code() != type_.ptype_info->hash_code()) {
    LOG(FATAL) << "Could not cast type " << type_.ptype_info->name() << " to type "
               << typeid(T).name();
    return false;
  }
  return true;
}

template<typename T>
const T& Any::Cast() const {
  CheckType<T>();
  return *reinterpret_cast<const T*>(data_->Ptr());
}

template<typename T>
T& Any::Cast() {
  CheckType<T>();
  return *const_cast<T*>(reinterpret_cast<const T*>(data_->Ptr()));
}

template<typename T>
const T& any_cast(const Any& any) {
  return any.Cast<T>();
}

template<typename T>
T& any_cast(Any& any) {
  return any.Cast<T>();
}

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_ANY_H_
