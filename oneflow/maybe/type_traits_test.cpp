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

#include <gtest/gtest.h>
#include <type_traits>
#include "oneflow/maybe/type_traits.h"

using namespace oneflow::maybe;

TEST(TypeTraits, Basics) {
  static_assert(Conj<std::true_type, std::true_type>, "");
  static_assert(!Conj<std::false_type, std::true_type>, "");
  static_assert(!Conj<std::true_type, std::false_type>, "");
  static_assert(!Conj<std::false_type, std::false_type>, "");
  static_assert(!Conj<std::true_type, std::true_type, std::false_type>, "");

  static_assert(Disj<std::true_type, std::true_type>, "");
  static_assert(Disj<std::false_type, std::true_type>, "");
  static_assert(Disj<std::true_type, std::false_type>, "");
  static_assert(!Disj<std::false_type, std::false_type>, "");
  static_assert(Disj<std::true_type, std::true_type, std::false_type>, "");
  static_assert(!Disj<std::false_type, std::false_type, std::false_type>, "");

  static_assert(std::is_same<TypeGet<0, int>, int>::value, "");
  static_assert(std::is_same<TypeGet<0, int, float>, int>::value, "");
  static_assert(std::is_same<TypeGet<1, int, float>, float>::value, "");
  static_assert(std::is_same<TypeGet<2, int, float, bool, int>, bool>::value, "");
  static_assert(std::is_same<TypeGet<2, int, int>, TypeNotFound>::value, "");
  static_assert(std::is_same<TypeGet<2, int, int, float>, float>::value, "");
  static_assert(std::is_same<TypeGet<2, int, int, float, int>, float>::value, "");
  static_assert(std::is_same<TypeGet<2>, TypeNotFound>::value, "");

  static_assert(IndexGet<int, int> == 0, "");
  static_assert(IndexGet<int, float> == NPos, "");
  static_assert(IndexGet<int, int, int> == 0, "");
  static_assert(IndexGet<int, float, int> == 1, "");
  static_assert(IndexGet<bool, int, float, int, bool, bool, int> == 3, "");
  static_assert(IndexGet<int> == NPos, "");

  static_assert(!TypeIn<int>, "");
  static_assert(TypeIn<int, int>, "");
  static_assert(TypeIn<int, float, int>, "");
  static_assert(!TypeIn<int, float, float, bool>, "");
  static_assert(TypeIn<int, float, bool, int, float>, "");
  static_assert(TypeIn<bool, float, float, bool>, "");

  static_assert(IsDifferentTypes<int, float, bool>, "");
  static_assert(!IsDifferentTypes<int, float, int, bool>, "");
  static_assert(IsDifferentTypes<int>, "");
  static_assert(!IsDifferentTypes<int, int>, "");
}
