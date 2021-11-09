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

#ifndef ONEFLOW_MAYBE_TYPE_TRAITS_H_
#define ONEFLOW_MAYBE_TYPE_TRAITS_H_

#include <cstddef>
#include <type_traits>
#include <tuple>
#include <utility>

namespace oneflow {

namespace maybe {

template<bool B>
using BoolConstant = std::integral_constant<bool, B>;

template<std::size_t I>
using IndexConstant = std::integral_constant<std::size_t, I>;

constexpr std::size_t NPos = -1;

template<typename...>
struct ConjT : std::true_type {};
template<typename B1>
struct ConjT<B1> : B1 {};
template<typename B1, typename... Bn>
struct ConjT<B1, Bn...> : std::conditional_t<bool(B1::value), ConjT<Bn...>, B1> {};

template<typename... B>
constexpr bool Conj = ConjT<B...>::value;

template<typename...>
struct DisjT : std::false_type {};
template<typename B1>
struct DisjT<B1> : B1 {};
template<typename B1, typename... Bn>
struct DisjT<B1, Bn...> : std::conditional_t<bool(B1::value), B1, DisjT<Bn...>> {};

template<typename... B>
constexpr bool Disj = DisjT<B...>::value;

template<typename B>
struct NegT : BoolConstant<!bool(B::value)> {};

template<typename B>
constexpr bool Neg = NegT<B>::value;

struct TypeNotFound;

// return TypeNotFound while out of range
template<std::size_t I, typename... Tn>
struct TypeGetT;

template<std::size_t I, typename T1, typename... Tn>
struct TypeGetT<I, T1, Tn...> : TypeGetT<I - 1, Tn...> {};

template<typename T1, typename... Tn>
struct TypeGetT<0, T1, Tn...> {
  using type = T1;
};

template<std::size_t N>
struct TypeGetT<N> {
  using type = TypeNotFound;
};

template<std::size_t I, typename... Ts>
using TypeGet = typename TypeGetT<I, Ts...>::type;

// return NPos (-1) while not found
template<std::size_t I, typename T, typename... Tn>
struct IndexGetT;

template<std::size_t I, typename T, typename T1, typename... Tn>
struct IndexGetT<I, T, T1, Tn...> : IndexGetT<I + 1, T, Tn...> {};

template<std::size_t I, typename T1, typename... Tn>
struct IndexGetT<I, T1, T1, Tn...> : IndexConstant<I> {};

template<std::size_t I, typename T>
struct IndexGetT<I, T> : IndexConstant<NPos> {};

template<typename T, typename... Ts>
constexpr auto IndexGet = IndexGetT<0, T, Ts...>::value;

template<typename T, typename... Ts>
constexpr auto TypeIn = IndexGet<T, Ts...> != NPos;

template<typename T, typename... Ts>
using TypeInT = BoolConstant<TypeIn<T, Ts...>>;

template<typename T>
struct RemoveCVRefT {
  using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template<typename T>
using RemoveCVRef = typename RemoveCVRefT<T>::type;

template<typename T, typename... Ts>
struct IsDifferentTypesT : BoolConstant<!TypeIn<T, Ts...> && IsDifferentTypesT<Ts...>::value> {};

template<typename T>
struct IsDifferentTypesT<T> : std::true_type {};

template<typename T, typename... Ts>
constexpr auto IsDifferentTypes = IsDifferentTypesT<T, Ts...>::value;

}  // namespace maybe

}  // namespace oneflow

#endif  // ONEFLOW_MAYBE_TYPE_TRAITS_H_
