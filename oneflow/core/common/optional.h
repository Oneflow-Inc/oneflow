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

#ifndef ONEFLOW_CORE_COMMON_OPTIONAL_H_
#define ONEFLOW_CORE_COMMON_OPTIONAL_H_

#include "oneflow/core/common/type_traits.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {
namespace internal {

template<typename T, typename U = void>
class Storage;

template<typename T>
class Storage<T, typename std::enable_if<IsScalarType<T>::value>::type> {
 public:
  Storage() : value_() {}

  template<typename... Args,
           typename std::enable_if<std::is_constructible<T, Args...>::value, int>::type = 0>
  Storage(Args&&... args) {
    new (&value_) T(std::forward<Args>(args)...);
  }

  Storage& operator=(const T& value) {
    value_ = value;
    return *this;
  }
  Storage& operator=(T&& value) {
    value_ = std::move(value);
    return *this;
  }
  Storage& operator=(const Storage<T>& rhs) {
    value_ = rhs.value_;
    return *this;
  }
  Storage& operator=(Storage<T>&& rhs) {
    value_ = std::move(rhs.value_);
    return *this;
  }

  Maybe<T> value() const { return value_; }

 private:
  T value_;
};

template<typename T>
class Storage<T, typename std::enable_if<!IsScalarType<T>::value>::type> {
 public:
  Storage() = default;

  template<typename... Args,
           typename std::enable_if<std::is_constructible<T, Args...>::value, int>::type = 0>
  Storage(Args&&... args) {
    value_ = std::make_shared<T>(std::forward<Args>(args)...);
  }

  Storage(const std::shared_ptr<T>& value) : value_(value) {}

  Storage& operator=(const T& value) {
    if (value_) {
      *value_ = value;
    } else {
      value_ = std::make_shared<T>(value);
    }
    return *this;
  }
  Storage& operator=(T&& value) {
    if (value_) {
      *value_ = std::move(value);
    } else {
      value_ = std::make_shared<T>(value);
    }
    return *this;
  }
  Storage& operator=(const Storage<T>& rhs) {
    value_ = rhs.value_;
    return *this;
  }
  Storage& operator=(Storage<T>&& rhs) {
    value_ = std::move(rhs.value_);
    return *this;
  }

  Maybe<T> value() const { return value_; }

 private:
  std::shared_ptr<T> value_;
};

}  // namespace internal

template<typename T>
class Optional final {
 private:
  template<typename U>
  using is_self = std::is_same<Optional, typename std::decay<U>::type>;

 public:
  Optional() : init_(false) {}

  template<typename U,
           typename std::enable_if<!is_self<U>::value
                                       && std::is_constructible<internal::Storage<T>, U>::value,
                                   int>::type = 0>
  Optional(U&& val) : init_(true), storage_(std::forward<U>(val)) {}

  ~Optional() = default;

  Optional(const Optional<T>& rhs) : init_(rhs.init_) {
    if (init_) { storage_ = rhs.storage_; }
  }

  Optional(Optional<T>&& rhs) : init_(rhs.init_) {
    if (init_) { storage_ = std::move(rhs.storage_); }
  }

  Optional& operator=(const T& val) {
    init_ = true;
    storage_ = val;
    return *this;
  }

  Optional& operator=(T&& val) {
    init_ = true;
    storage_ = std::move(val);
    return *this;
  }

  Optional& operator=(const Optional<T>& rhs) {
    init_ = rhs.init_;
    if (init_) { storage_ = rhs.storage_; }
    return *this;
  }

  Optional& operator=(Optional<T>&& rhs) {
    init_ = rhs.init_;
    if (init_) { storage_ = std::move(rhs.storage_); }
    return *this;
  }

  Maybe<T> value() const {
    CHECK_OR_RETURN(has_value()) << "Optional has no value.";
    return storage_.value();
  }

  bool has_value() const { return init_; }
  operator bool() const { return has_value(); }

 private:
  bool init_;
  internal::Storage<T> storage_;
};

template<typename T>
class Optional<T&> final {
 public:
  Optional() : value_ptr_(nullptr) {}

  Optional(T& val) : value_ptr_(&val) {}

  ~Optional() = default;

  Optional& operator=(const Optional<T&>& rhs) {
    value_ptr_ = rhs.value_ptr_;
    return *this;
  }

  Maybe<T&> value() const {
    CHECK_OR_RETURN(has_value()) << "Optional has no value.";
    return *value_ptr_;
  }

  bool has_value() const { return value_ptr_ != nullptr; }
  operator bool() const { return has_value(); }

 private:
  T* value_ptr_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_OPTIONAL_H_
