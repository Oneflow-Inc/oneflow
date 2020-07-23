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
#ifndef ONEFLOW_CORE_COMMON_PREPROCESSOR_H_
#define ONEFLOW_CORE_COMMON_PREPROCESSOR_H_

#include "oneflow/core/common/preprocessor_internal.h"

// basic
#define OF_PP_CAT(a, b) OF_PP_INTERNAL_CAT(a, b)

#define OF_PP_STRINGIZE(x) OF_PP_INTERNAL_STRINGIZE(x)

#define OF_PP_PAIR_FIRST(pair) OF_PP_INTERNAL_PAIR_FIRST(pair)

#define OF_PP_PAIR_SECOND(pair) OF_PP_INTERNAL_PAIR_SECOND(pair)

#define OF_PP_TUPLE_SIZE(t) OF_PP_INTERNAL_TUPLE_SIZE(t)

#define OF_PP_TUPLE_ELEM(n, t) OF_PP_INTERNAL_TUPLE_ELEM(n, t)

#define OF_PP_MAKE_TUPLE_SEQ(...) OF_PP_INTERNAL_MAKE_TUPLE_SEQ(__VA_ARGS__)

#define OF_PP_FOR_EACH_TUPLE(macro, seq) OF_PP_INTERNAL_FOR_EACH_TUPLE(macro, seq)

#define OF_PP_OUTTER_FOR_EACH_TUPLE(macro, seq) OF_PP_INTERNAL_OUTTER_FOR_EACH_TUPLE(macro, seq)

#define OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(macro, ...) \
  OF_PP_INTERNAL_SEQ_PRODUCT_FOR_EACH_TUPLE(macro, __VA_ARGS__)

// advanced

#define OF_PP_VARIADIC_SIZE(...) OF_PP_INTERNAL_VARIADIC_SIZE(__VA_ARGS__)

#define OF_PP_SEQ_SIZE(seq) OF_PP_INTERNAL_SEQ_SIZE(seq)

#define OF_PP_ATOMIC_TO_TUPLE(x) (x)

#define OF_PP_FOR_EACH_ATOMIC(macro, seq) \
  OF_PP_FOR_EACH_TUPLE(macro, OF_PP_SEQ_MAP(OF_PP_ATOMIC_TO_TUPLE, seq))

#define OF_PP_SEQ_PRODUCT(seq0, ...) OF_PP_INTERNAL_SEQ_PRODUCT(seq0, __VA_ARGS__)

#define OF_PP_SEQ_MAP(macro, seq) \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OF_PP_I_SEQ_MAP_DO_EACH, (macro), seq)
#define OF_PP_I_SEQ_MAP_DO_EACH(macro, elem) (macro(elem))

#define OF_PP_JOIN(glue, ...) OF_PP_INTERNAL_JOIN(glue, __VA_ARGS__)

#define OF_PP_TUPLE_PUSH_FRONT(t, x) OF_PP_INTERNAL_TUPLE_PUSH_FRONT(t, x)

#define OF_PP_FORCE(...) OF_PP_TUPLE2VARADIC(OF_PP_CAT((__VA_ARGS__), ))

#endif  // ONEFLOW_CORE_COMMON_PREPROCESSOR_H_
