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

#ifndef ONEFLOW_MAYBE_VARIANT_H_
#define ONEFLOW_MAYBE_VARIANT_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <functional>
#include <iostream>

#include "oneflow/maybe/utility.h"
#include "oneflow/maybe/type_traits.h"

namespace oneflow {

namespace maybe {

template<typename... Ts>
struct Variant;

namespace details {

// there are generally two ways to implement visit (like std::visit in c++17)
// 1. O(N) or O(log N), to iterate for all types or do a binary search on type index recursively
// 2. O(1), to store an static (storage duration) array of function pointers for every (Variant, F)
// where N = Variant<T...>::Num, and normally (in most cases) within the range [2, 5]
// the 2nd method is required in std::visit(f, x...) while sizeof...(x) == 1
// but weakness of the 2nd method is that compilers usually cannot efficiently optimize these
// function pointers (compared to trivial recursion, which is easy to do optimization, and also
// friendly to CPU cache) here we implement visit via the first method:
// 1. for 2 <= N < 4, we use the O(N) algorithm (TrivialRecursiveVisitImpl) for better optimization
// 2. for N >= 4, we use the O(log N) algorithm (BinarySearchVisitImpl) for less recursion rounds

struct VariantPrivateScope {
  template<typename R, typename F, typename V>
  static R TrivialRecursiveVisitImpl(F&& f, V&& v, InPlaceIndexT<RemoveCVRef<V>::Num - 1>) {
    // assume v.Index() == N - 1 now
    return static_cast<R>(
        std::forward<F>(f)(std::forward<V>(v).template Value<RemoveCVRef<V>::Num - 1>()));
  }

  template<typename R, std::size_t I, typename F, typename V,
           std::enable_if_t<(I < RemoveCVRef<V>::Num - 1), int> = 0>
  static R TrivialRecursiveVisitImpl(F&& f, V&& v, InPlaceIndexT<I>) {
    if (v.Index() == I) {
      return static_cast<R>(std::forward<F>(f)(std::forward<V>(v).template Value<I>()));
    }

    return TrivialRecursiveVisitImpl<R>(std::forward<F>(f), std::forward<V>(v),
                                        InPlaceIndex<I + 1>);
  }

  template<typename R, std::size_t I, typename F, typename V,
           std::enable_if_t<(I < RemoveCVRef<V>::Num), int> = 0>
  static R BinarySearchVisitImpl(F&& f, V&& v, InPlaceIndexT<I>, InPlaceIndexT<I>) {
    return static_cast<R>(std::forward<F>(f)(std::forward<V>(v).template Value<I>()));
  }

  template<typename R, std::size_t I, typename F, typename V,
           std::enable_if_t<(I + 1 < RemoveCVRef<V>::Num), int> = 0>
  static R BinarySearchVisitImpl(F&& f, V&& v, InPlaceIndexT<I>, InPlaceIndexT<I + 1>) {
    constexpr std::size_t M = (I + I + 1) / 2;
    constexpr std::size_t N = (M == I) ? I + 1 : I;

    if (v.Index() == M) {
      return static_cast<R>(std::forward<F>(f)(std::forward<V>(v).template Value<M>()));
    } else {
      return static_cast<R>(std::forward<F>(f)(std::forward<V>(v).template Value<N>()));
    }
  }

  template<typename R, std::size_t L, std::size_t U, typename F, typename V,
           std::enable_if_t<(L + 1 < U) && (U < RemoveCVRef<V>::Num), int> = 0>
  static R BinarySearchVisitImpl(F&& f, V&& v, InPlaceIndexT<L>, InPlaceIndexT<U>) {
    constexpr std::size_t M = (L + U) / 2;

    if (v.Index() < M) {
      return BinarySearchVisitImpl<R>(std::forward<F>(f), std::forward<V>(v), InPlaceIndex<L>,
                                      InPlaceIndex<M - 1>);
    } else if (v.Index() > M) {
      return BinarySearchVisitImpl<R>(std::forward<F>(f), std::forward<V>(v), InPlaceIndex<M + 1>,
                                      InPlaceIndex<U>);
    } else {
      return static_cast<R>(std::forward<F>(f)(std::forward<V>(v).template Value<M>()));
    }
  }

  template<typename R, typename F, typename V,
           std::enable_if_t<RemoveCVRef<V>::Num<4, int> = 0> static R VisitImpl(F&& f, V&& v) {
    return TrivialRecursiveVisitImpl<R>(std::forward<F>(f), std::forward<V>(v), InPlaceIndex<0>);
  }

  template<typename R, typename F, typename V, std::enable_if_t<RemoveCVRef<V>::Num >= 4, int> = 0>
  static R VisitImpl(F&& f, V&& v) {
    return BinarySearchVisitImpl<R>(std::forward<F>(f), std::forward<V>(v), InPlaceIndex<0>,
                                    InPlaceIndex<RemoveCVRef<V>::Num - 1>);
  }
};

struct AutoDeducedResultType;

template<typename R, typename F, typename... Ts>
struct VisitResultS {
  using type = R;
};

template<typename F, typename... Ts>
struct VisitResultS<AutoDeducedResultType, F, Ts...> {
  using type = std::common_type_t<decltype(std::declval<F>()(std::declval<Ts>()))...>;
};

template<typename R, typename F, typename... Ts>
using VisitResult = typename VisitResultS<R, F, Ts...>::type;

}  // namespace details

// preconditions: template type arguments must be no less than 2 different type
// and without reference and cv qualifiers
// this Variant DO NOT guarantee exception safety
template<typename... Ts>
struct Variant {  // NOLINT(cppcoreguidelines-pro-type-member-init)
 public:
  static_assert(sizeof...(Ts) > 1, "expected more than two types");
  static_assert(Conj<NegS<std::is_reference<Ts>>...>, "reference types are not allowed here");
  static_assert(Conj<NegS<DisjS<std::is_const<Ts>, std::is_volatile<Ts>>>...>,
                "cv qualifiers are not allowed here");
  // important precondition to optimize Visit via binary search
  static_assert(IsDifferentTypes<Ts...>, "expected all of different types");

  static constexpr std::size_t Num = sizeof...(Ts);

  template<typename T>
  static constexpr std::size_t IndexOfType = IndexGet<T, Ts...>;

  template<typename T>
  static constexpr bool HasType = TypeIn<T, Ts...>;

  template<std::size_t I>
  using TypeByIndex = TypeGet<I, Ts...>;

  template<typename T = TypeByIndex<0>,
           std::enable_if_t<std::is_default_constructible<T>::value, int> = 0>
  Variant() {  // NOLINT(cppcoreguidelines-pro-type-member-init)
    Construct<0>();
  }

  // unlike std::variant, we only accept exact types to avoid wrong construction
  template<typename T, std::enable_if_t<HasType<RemoveCVRef<T>>, int> = 0>
  Variant(T&& v) {  // NOLINT(cppcoreguidelines-pro-type-member-init, google-explicit-constructor)
    Construct<RemoveCVRef<T>>(std::forward<T>(v));
  }

  template<typename T, typename... Args, std::enable_if_t<HasType<RemoveCVRef<T>>, int> = 0>
  explicit Variant(InPlaceTypeT<T>,  // NOLINT(cppcoreguidelines-pro-type-member-init)
                   Args&&... args) {
    Construct<RemoveCVRef<T>>(std::forward<Args>(args)...);
  }

  template<std::size_t I, typename... Args, std::enable_if_t<(I < Num), int> = 0>
  explicit Variant(InPlaceIndexT<I>,  // NOLINT(cppcoreguidelines-pro-type-member-init)
                   Args&&... args) {
    Construct<I>(std::forward<Args>(args)...);
  }

  template<typename R = details::AutoDeducedResultType, typename F>
  decltype(auto) Visit(F&& f) & {
    using Result = details::VisitResult<R, F, Ts&...>;
    return details::VariantPrivateScope::VisitImpl<Result>(std::forward<F>(f), *this);
  }

  template<typename R = details::AutoDeducedResultType, typename F>
  decltype(auto) Visit(F&& f) && {
    using Result = details::VisitResult<R, F, Ts&&...>;
    return details::VariantPrivateScope::VisitImpl<Result>(std::forward<F>(f), std::move(*this));
  }

  template<typename R = details::AutoDeducedResultType, typename F>
  decltype(auto) Visit(F&& f) const& {
    using Result = details::VisitResult<R, F, const Ts&...>;
    return details::VariantPrivateScope::VisitImpl<Result>(std::forward<F>(f), *this);
  }

  Variant(const Variant& v) {  // NOLINT(cppcoreguidelines-pro-type-member-init)
    CopyConstruct(v);
  }

  Variant(Variant&& v) noexcept {  // NOLINT(cppcoreguidelines-pro-type-member-init)
    CopyConstruct(std::move(v));
  }

  template<typename T, std::enable_if_t<HasType<RemoveCVRef<T>>, int> = 0>
  Variant& operator=(T&& v) {
    using Type = RemoveCVRef<T>;

    Emplace<Type>(std::forward<T>(v));

    return *this;
  }

  Variant& operator=(const Variant& v) {
    Copy(v);
    return *this;
  }

  Variant& operator=(Variant&& v) noexcept {
    Copy(std::move(v));
    return *this;
  }

  std::size_t Index() const { return type_index_; }

  template<typename T, std::enable_if_t<HasType<T>, int> = 0>
  bool Is() const {
    return type_index_ == IndexOfType<T>;
  }

  ~Variant() { Destory(); }

  bool operator==(const Variant& v) const {
    if (type_index_ != v.type_index_) return false;

    return v.Visit(
        [this](const auto& elem) { return elem == Value<RemoveCVRef<decltype(elem)>>(); });
  }

  bool operator!=(const Variant& v) const { return !operator==(v); }

  bool operator<(const Variant& v) const {
    if (type_index_ < v.type_index_) return true;
    if (type_index_ > v.type_index_) return false;

    return v.Visit(
        [this](const auto& elem) { return Value<RemoveCVRef<decltype(elem)>>() < elem; });
  }

  bool operator>=(const Variant& v) const { return !(*this < v); }

  bool operator>(const Variant& v) const {
    if (type_index_ > v.type_index_) return true;
    if (type_index_ < v.type_index_) return false;

    return v.Visit(
        [this](const auto& elem) { return Value<RemoveCVRef<decltype(elem)>>() > elem; });
  }

  bool operator<=(const Variant& v) const { return !(*this > v); }

  template<typename T, std::enable_if_t<HasType<T>, int> = 0>
  friend bool operator==(const Variant& v, const T& x) {
    if (v.type_index_ != IndexOfType<T>) return false;

    return v.Value<T>() == x;
  }

  template<typename T, std::enable_if_t<HasType<T>, int> = 0>
  friend bool operator!=(const Variant& v, const T& x) {
    return !(v == x);
  }

  template<typename T, std::enable_if_t<HasType<T>, int> = 0>
  friend bool operator==(const T& x, const Variant& v) {
    return v == x;
  }

  template<typename T, std::enable_if_t<HasType<T>, int> = 0>
  friend bool operator!=(const T& x, const Variant& v) {
    return !(v == x);
  }

  template<typename T, typename... Args>
  T& Emplace(Args&&... args) {
    if (Is<T>()) {
      return Value<T>() = T(std::forward<Args>(args)...);
    } else {
      Destory();
      Construct<T>(std::forward<Args>(args)...);
      return Value<T>();
    }
  }

  template<std::size_t I, typename... Args>
  decltype(auto) Emplace(Args&&... args) {
    return Emplace<TypeByIndex<I>>(std::forward<Args>(args)...);
  }

  template<typename T, std::enable_if_t<HasType<T>, int> = 0>
  T& Get() & {
    OF_MAYBE_ASSERT_EQ(Index(), IndexOfType<T>);
    return Value<T>();
  }

  template<typename T, std::enable_if_t<HasType<T>, int> = 0>
  T&& Get() && {
    OF_MAYBE_ASSERT_EQ(Index(), IndexOfType<T>);
    return std::move(*this).template Value<T>();
  }

  template<typename T, std::enable_if_t<HasType<T>, int> = 0>
  const T& Get() const& {
    OF_MAYBE_ASSERT_EQ(Index(), IndexOfType<T>);
    return Value<T>();
  }

  template<std::size_t I, std::enable_if_t<(I < Num), int> = 0>
  TypeByIndex<I>& Get() & {
    OF_MAYBE_ASSERT_EQ(Index(), I);
    return Value<I>();
  }

  template<std::size_t I, std::enable_if_t<(I < Num), int> = 0>
  TypeByIndex<I>&& Get() && {
    OF_MAYBE_ASSERT_EQ(Index(), I);
    return std::move(*this).template Value<I>();
  }

  template<std::size_t I, std::enable_if_t<(I < Num), int> = 0>
  const TypeByIndex<I>& Get() const& {
    OF_MAYBE_ASSERT_EQ(Index(), I);
    return Value<I>();
  }

 protected:
  // use std::launder while updating to c++17
  template<typename T, std::enable_if_t<HasType<T>, int> = 0>
  T& Value() & {
    return *reinterpret_cast<T*>(storage_);
  }

  template<typename T, std::enable_if_t<HasType<T>, int> = 0>
  T&& Value() && {
    return std::move(*reinterpret_cast<T*>(storage_));
  }

  template<typename T, std::enable_if_t<HasType<T>, int> = 0>
  const T& Value() const& {
    return *reinterpret_cast<const T*>(storage_);
  }

  template<std::size_t I, std::enable_if_t<(I < Num), int> = 0>
  TypeByIndex<I>& Value() & {
    return *reinterpret_cast<TypeByIndex<I>*>(storage_);
  }

  template<std::size_t I, std::enable_if_t<(I < Num), int> = 0>
  TypeByIndex<I>&& Value() && {
    return std::move(*reinterpret_cast<TypeByIndex<I>*>(storage_));
  }

  template<std::size_t I, std::enable_if_t<(I < Num), int> = 0>
  const TypeByIndex<I>& Value() const& {
    return *reinterpret_cast<const TypeByIndex<I>*>(storage_);
  }

 private:
  static constexpr const std::size_t size = std::max({sizeof(Ts)...});

  alignas(Ts...) unsigned char storage_[size];
  std::uint8_t type_index_;

  friend struct details::VariantPrivateScope;

  template<typename T, typename... Args, std::enable_if_t<HasType<T> && IsAggregate<T>, int> = 0>
  void Construct(Args&&... args) {
    new (storage_) T{std::forward<Args>(args)...};
    type_index_ = IndexOfType<T>;
  }

  template<typename T, typename... Args, std::enable_if_t<HasType<T> && !IsAggregate<T>, int> = 0>
  void Construct(Args&&... args) {
    new (storage_) T(std::forward<Args>(args)...);
    type_index_ = IndexOfType<T>;
  }

  template<std::size_t I, typename... Args, std::enable_if_t<(I < Num), int> = 0>
  void Construct(Args&&... args) {
    Construct<TypeByIndex<I>>(std::forward<Args>(args)...);
  }

  template<typename V>
  void CopyConstruct(V&& v) {
    std::forward<V>(v).Visit([this](auto&& elem) {
      using T = RemoveCVRef<decltype(elem)>;

      new (storage_) T(std::forward<decltype(elem)>(elem));
      type_index_ = IndexOfType<T>;
    });
  }

  template<typename V>
  void Copy(V&& v) {
    std::forward<V>(v).Visit([this](auto&& elem) {
      using T = RemoveCVRef<decltype(elem)>;

      if (Is<T>()) {
        Value<T>() = std::forward<decltype(elem)>(elem);
      } else {
        Destory();
        Construct<T>(std::forward<decltype(elem)>(elem));
      }
    });
  }

  void Destory() {
    Visit([this](auto& elem) {
      using T = RemoveCVRef<decltype(elem)>;

      Value<T>().~T();
    });
  }
};

template<typename... Ts>
using OptionalVariant = Variant<NullOptType, Ts...>;

}  // namespace maybe

}  // namespace oneflow

namespace std {

template<typename... Ts>
struct hash<oneflow::maybe::Variant<Ts...>> {
  size_t operator()(const oneflow::maybe::Variant<Ts...>& v) const noexcept {
    size_t seed = hash<size_t>()(v.Index());

    v.Visit([&seed](const auto& x) {
      using type = oneflow::maybe::RemoveCVRef<decltype(x)>;
      oneflow::maybe::HashCombine<type>(seed, x);
    });

    return seed;
  }
};

}  // namespace std

#endif  // ONEFLOW_MAYBE_VARIANT_H_
