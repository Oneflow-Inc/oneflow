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
#include "config.h"

namespace oneflow {

namespace maybe {

// in this file, xxxS represents struct of xxx
// for implementant aspect, xxx is an alias of xxxS::type or xxxS::value

template<bool B>
using BoolConstant = std::integral_constant<bool, B>;

template<std::size_t I>
using IndexConstant = std::integral_constant<std::size_t, I>;

constexpr std::size_t NPos = -1;

template<typename...>
struct ConjS : std::true_type {};
template<typename B1>
struct ConjS<B1> : B1 {};
template<typename B1, typename... Bn>
struct ConjS<B1, Bn...> : std::conditional_t<bool(B1::value), ConjS<Bn...>, B1> {};

template<typename... B>
constexpr bool Conj = ConjS<B...>::value;

template<typename...>
struct DisjS : std::false_type {};
template<typename B1>
struct DisjS<B1> : B1 {};
template<typename B1, typename... Bn>
struct DisjS<B1, Bn...> : std::conditional_t<bool(B1::value), B1, DisjS<Bn...>> {};

template<typename... B>
constexpr bool Disj = DisjS<B...>::value;

template<typename B>
struct NegS : BoolConstant<!bool(B::value)> {};

template<typename B>
constexpr bool Neg = NegS<B>::value;

struct TypeNotFound;

// return TypeNotFound while out of range
template<std::size_t I, typename... Tn>
struct TypeGetS;

template<std::size_t I, typename T1, typename... Tn>
struct TypeGetS<I, T1, Tn...> : TypeGetS<I - 1, Tn...> {};

template<typename T1, typename... Tn>
struct TypeGetS<0, T1, Tn...> {
  using type = T1;
};

template<std::size_t N>
struct TypeGetS<N> {
  using type = TypeNotFound;
};

template<std::size_t I, typename... Ts>
using TypeGet = typename TypeGetS<I, Ts...>::type;

// return NPos (-1) while not found
template<std::size_t I, typename T, typename... Tn>
struct IndexGetFromS;

template<std::size_t I, typename T, typename T1, typename... Tn>
struct IndexGetFromS<I, T, T1, Tn...> : IndexGetFromS<I + 1, T, Tn...> {};

template<std::size_t I, typename T1, typename... Tn>
struct IndexGetFromS<I, T1, T1, Tn...> : IndexConstant<I> {};

template<std::size_t I, typename T>
struct IndexGetFromS<I, T> : IndexConstant<NPos> {};

template<typename T, typename... Ts>
constexpr auto IndexGet = IndexGetFromS<0, T, Ts...>::value;

template<typename T, typename... Ts>
constexpr auto TypeIn = IndexGet<T, Ts...> != NPos;

template<typename T, typename... Ts>
using TypeInS = BoolConstant<TypeIn<T, Ts...>>;

template<typename T>
struct RemoveCVRefS {
  using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template<typename T>
using RemoveCVRef = typename RemoveCVRefS<T>::type;

template<typename T, typename... Ts>
struct IsDifferentTypesS : BoolConstant<!TypeIn<T, Ts...> && IsDifferentTypesS<Ts...>::value> {};

template<typename T>
struct IsDifferentTypesS<T> : std::true_type {};

template<typename T, typename... Ts>
constexpr auto IsDifferentTypes = IsDifferentTypesS<T, Ts...>::value;

template<typename T>
struct ConstRefExceptVoidS {
  using type = const T&;
};

template<>
struct ConstRefExceptVoidS<void> {
  using type = void;
};

template<typename T>
using ConstRefExceptVoid = typename ConstRefExceptVoidS<T>::type;

template<typename T>
using RemoveRValRef =
    std::conditional_t<std::is_rvalue_reference<T>::value, std::remove_reference_t<T>, T>;

template<typename T>
constexpr bool IsAggregate = OF_MAYBE_IS_AGGREGATE(T);

}  // namespace maybe

}  // namespace oneflow

#endif  // ONEFLOW_MAYBE_TYPE_TRAITS_H_
