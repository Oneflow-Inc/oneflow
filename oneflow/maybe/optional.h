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

#ifndef ONEFLOW_MAYBE_OPTIONAL_H_
#define ONEFLOW_MAYBE_OPTIONAL_H_

#include <type_traits>
#include <utility>

#include "oneflow/maybe/utility.h"
#include "oneflow/maybe/type_traits.h"

namespace oneflow {

namespace maybe {

namespace details {

template<typename T, typename = void>
struct OptionalStorage {
 private:
  bool has;
  alignas(T) char value[sizeof(T)];

  using Type = std::remove_const_t<T>;

 public:
  void Init() { has = false; }

  T& Value() & { return *reinterpret_cast<T*>(value); }

  Type&& Value() && { return std::move(*const_cast<Type*>(reinterpret_cast<T*>(value))); }

  const T& Value() const& { return *reinterpret_cast<const T*>(value); }

  bool HasValue() const { return has; }

  void Reset() {
    if (has) {
      has = false;
      Value().~T();
    }
  }

  void Destory() {
    if (has) { Value().~T(); }
  }

  template<typename... Args>
  void Construct(Args&&... args) {
    new (value) Type(std::forward<Args>(args)...);
    has = true;
  }

  template<typename... Args, typename U = T, std::enable_if_t<!std::is_const<U>::value, int> = 0>
  T& Emplace(Args&&... args) {
    if (!has) {
      Construct(std::forward<Args>(args)...);
      return Value();
    } else {
      return Value() = Type(std::forward<Args>(args)...);
    }
  }

  template<typename... Args, typename U = T, std::enable_if_t<std::is_const<U>::value, int> = 0>
  T& Emplace(Args&&... args) {
    Destory();
    Construct(std::forward<Args>(args)...);
    return Value();
  }

  template<typename OS>
  void CopyConstruct(OS&& s) {
    has = s.has;

    if (has) { new (value) Type(std::forward<OS>(s).Value()); }
  }

  template<typename OS>
  void Copy(OS&& s) {
    if (s.has) {
      Emplace(std::forward<OS>(s).Value());
    } else {
      Reset();
    }
  }
};

template<typename T>
struct OptionalStorage<T, std::enable_if_t<std::is_scalar<T>::value>> {
 private:
  using Type = std::remove_const_t<T>;

  bool has;
  Type value;

 public:
  void Init() { has = false; }

  T& Value() & { return value; }

  Type&& Value() && { return std::move(const_cast<Type&>(value)); }

  const T& Value() const& { return value; }

  bool HasValue() const { return has; }

  void Reset() { has = false; }

  void Destory() {}

  template<typename U>
  void Construct(const U& v) {
    value = v;
    has = true;
  }

  template<typename U>
  T& Emplace(const U& v) {
    Construct(v);
    return Value();
  }

  void CopyConstruct(const OptionalStorage& s) {
    has = s.has;
    value = s.value;
  }

  void Copy(const OptionalStorage& s) { CopyConstruct(s); }
};

template<typename T>
struct OptionalStorage<T, std::enable_if_t<std::is_reference<T>::value>> {
  static_assert(std::is_lvalue_reference<T>::value, "rvalue reference is not allowed here");

  using Type = std::remove_reference_t<T>;

 private:
  Type* value;

 public:
  void Init() { value = nullptr; }

  T Value() { return *value; }

  const Type& Value() const { return *value; }

  bool HasValue() const { return value != nullptr; }

  void Reset() { value = nullptr; }

  void Destory() {}

  void Construct(T v) { value = &v; }

  T Emplace(T v) {
    Construct(v);
    return Value();
  }

  void CopyConstruct(const OptionalStorage& s) { value = s.value; }

  void Copy(const OptionalStorage& s) { CopyConstruct(s); }
};

struct OptionalPrivateScope {
  template<typename T>
  static decltype(auto) Value(T&& opt) {
    return std::forward<T>(opt).Value();
  }
};

}  // namespace details

// this Optional DO NOT guarantee exception safty
template<typename T>
struct Optional {
 private:
  details::OptionalStorage<T> storage;

  using Type = std::remove_const_t<T>;

  decltype(auto) Value() & { return storage.Value(); }

  decltype(auto) Value() && { return std::move(storage).Value(); }

  decltype(auto) Value() const& { return storage.Value(); }

  friend struct details::OptionalPrivateScope;

 public:
  using ValueType = T;

  explicit Optional() { storage.Init(); };

  Optional(NullOptType) { storage.Init(); }  // NOLINT(google-explicit-constructor)

  Optional(const T& v) { storage.Construct(v); }  // NOLINT(google-explicit-constructor)

  template<typename U = T, std::enable_if_t<!std::is_reference<U>::value, int> = 0>
  Optional(Type&& v) {  // NOLINT(google-explicit-constructor)
    storage.Construct(std::move(v));
  }

  Optional(const Optional& opt) { storage.CopyConstruct(opt.storage); }
  Optional(Optional&& opt) noexcept { storage.CopyConstruct(std::move(opt.storage)); }

  template<typename... Args>
  explicit Optional(InPlaceT, Args&&... args) {
    storage.Construct(std::forward<Args>(args)...);
  }

  ~Optional() { storage.Destory(); }

  Optional& operator=(NullOptType) {
    storage.Reset();
    return *this;
  }

  Optional& operator=(const T& v) {
    storage.Emplace(v);
    return *this;
  }

  template<typename U = T, std::enable_if_t<!std::is_reference<U>::value, int> = 0>
  Optional& operator=(Type&& v) {
    storage.Emplace(std::move(v));
    return *this;
  }

  template<typename... Args>
  decltype(auto) Emplace(Args&&... args) {
    return storage.Emplace(std::forward<Args>(args)...);
  }

  Optional& operator=(const Optional& opt) {
    storage.Copy(opt.storage);
    return *this;
  }

  Optional& operator=(Optional&& opt) noexcept {
    storage.Copy(std::move(opt.storage));
    return *this;
  }

  bool HasValue() const { return storage.HasValue(); }
  explicit operator bool() const { return HasValue(); }

  bool operator==(const Optional& opt) const {
    if (HasValue()) {
      if (opt.HasValue()) {
        return Value() == opt.Value();
      } else {
        return false;
      }
    } else {
      return !opt.HasValue();
    }
  }

  bool operator!=(const Optional& opt) const { return !operator==(opt); }

  decltype(auto) ValueOr(const T& v) const& {
    if (HasValue()) {
      return Value();
    } else {
      return v;
    }
  }

  template<typename U = T, std::enable_if_t<!std::is_reference<U>::value, int> = 0>
  auto ValueOr(T&& v) const& {
    if (HasValue()) {
      return Value();
    } else {
      return std::move(v);
    }
  }

  template<typename U = T, std::enable_if_t<!std::is_reference<U>::value, int> = 0>
  auto ValueOr(const T& v) && {
    if (HasValue()) {
      return std::move(*this).Value();
    } else {
      return v;
    }
  }

  template<typename U = T, std::enable_if_t<!std::is_reference<U>::value, int> = 0>
  decltype(auto) ValueOr(T&& v) && {
    if (HasValue()) {
      return std::move(*this).Value();
    } else {
      return std::move(v);
    }
  }

  void Reset() { storage.Reset(); }
};

}  // namespace maybe

}  // namespace oneflow

#endif  // ONEFLOW_MAYBE_OPTIONAL_H_
