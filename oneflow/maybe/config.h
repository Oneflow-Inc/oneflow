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

#ifndef ONEFLOW_MAYBE_CONFIG_H_
#define ONEFLOW_MAYBE_CONFIG_H_

#include <cassert>

// pre-define it if you use a logging library like glog
#ifndef OF_MAYBE_ASSERT
#define OF_MAYBE_ASSERT(_cond_) assert(_cond_)
#endif

// ASSERT_EQ is different from ASSERT in logging / testing framework
// pre-define it if you use a logging library like glog
#ifndef OF_MAYBE_ASSERT_EQ
#define OF_MAYBE_ASSERT_EQ(_lhs_, _rhs_) OF_MAYBE_ASSERT(_lhs_ == _rhs_)
#endif

#if __GNUC__ >= 7
#define OF_MAYBE_HAS_IS_AGGREGATE
// in old versions of clang, __has_builtin(__is_aggregate) returns false
#elif __clang__
#if !__is_identifier(__is_aggregate)
#define OF_MAYBE_HAS_IS_AGGREGATE
#endif
#else
#if __has_builtin(__is_aggregate)
#define OF_MAYBE_HAS_IS_AGGREGATE
#endif
#endif

#ifdef OF_MAYBE_HAS_IS_AGGREGATE
#define OF_MAYBE_IS_AGGREGATE(...) (__is_aggregate(__VA_ARGS__))
#else
// decay to POD checking if no such builtin (because implementing __is_aggregate need reflection)
#define OF_MAYBE_IS_AGGREGATE(...) \
  (std::is_standard_layout<__VA_ARGS__>::value && std::is_trivial<__VA_ARGS__>::value)
#endif

// `__builtin_expect` exists at least since GCC 4 / Clang 3
#define OF_MAYBE_EXPECT_FALSE(x) (__builtin_expect((x), 0))

#if __has_cpp_attribute(nodiscard)
#define OF_MAYBE_NODISCARD_FUNC [[nodiscard]]
#define OF_MAYBE_NODISCARD_TYPE [[nodiscard]]
#elif __has_attribute(warn_unused_result)
#define OF_MAYBE_NODISCARD_FUNC \
  __attribute__((warn_unused_result))  // or [[gnu::warn_unused_result]]
#define OF_MAYBE_NODISCARD_TYPE
#endif

#endif  // ONEFLOW_MAYBE_CONFIG_H_
