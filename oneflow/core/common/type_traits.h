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
#ifndef ONEFLOW_CORE_COMMON_TYPE_TRAITS_H_
#define ONEFLOW_CORE_COMMON_TYPE_TRAITS_H_
#include <type_traits>

namespace std {

#if __GNUG__ && __GNUC__ < 5 && !__clang__
// copied from
// https://llvm.org/doxygen/type__traits_8h_source.html
namespace detail {
/// Internal utility to detect trivial copy construction.
template<typename T>
union copy_construction_triviality_helper {
  T t;
  copy_construction_triviality_helper() = default;
  copy_construction_triviality_helper(const copy_construction_triviality_helper&) = default;
  ~copy_construction_triviality_helper() = default;
};
/// Internal utility to detect trivial move construction.
template<typename T>
union move_construction_triviality_helper {
  T t;
  move_construction_triviality_helper() = default;
  move_construction_triviality_helper(move_construction_triviality_helper&&) = default;
  ~move_construction_triviality_helper() = default;
};

template<class T>
union trivial_helper {
  T t;
};

}  // end namespace detail

// is_trivially_copyable
// An implementation of `std::is_trivially_copyable` since STL version
// is not equally supported by all compilers, especially GCC 4.8.
// Uniform implementation of this trait is important for ABI compatibility
// as it has an impact on SmallVector's ABI (among others).
template<typename T>
class is_trivially_copyable {
  // copy constructors
  static constexpr bool has_trivial_copy_constructor =
      std::is_copy_constructible<detail::trivial_helper<T>>::value;
  static constexpr bool has_deleted_copy_constructor = !std::is_copy_constructible<T>::value;

  // move constructors
  static constexpr bool has_trivial_move_constructor =
      std::is_move_constructible<detail::trivial_helper<T>>::value;
  static constexpr bool has_deleted_move_constructor = !std::is_move_constructible<T>::value;

  // copy assign
  static constexpr bool has_trivial_copy_assign =
      is_copy_assignable<detail::trivial_helper<T>>::value;
  static constexpr bool has_deleted_copy_assign = !is_copy_assignable<T>::value;

  // move assign
  static constexpr bool has_trivial_move_assign =
      is_move_assignable<detail::trivial_helper<T>>::value;
  static constexpr bool has_deleted_move_assign = !is_move_assignable<T>::value;

  // destructor
  static constexpr bool has_trivial_destructor =
      std::is_destructible<detail::trivial_helper<T>>::value;

 public:
  static constexpr bool value = has_trivial_destructor
                                && (has_deleted_move_assign || has_trivial_move_assign)
                                && (has_deleted_move_constructor || has_trivial_move_constructor)
                                && (has_deleted_copy_assign || has_trivial_copy_assign)
                                && (has_deleted_copy_constructor || has_trivial_copy_constructor);

#ifdef HAVE_STD_IS_TRIVIALLY_COPYABLE
  static_assert(
      value == std::is_trivially_copyable<T>::value,
      "inconsistent behavior between llvm:: and std:: implementation of is_trivially_copyable");
#endif
};
template<typename T>
class is_trivially_copyable<T*> : public true_type {};
#endif

}  // namespace std

namespace oneflow {

namespace detail {

template<typename T, typename Enabled = void>
struct ScalarOrConstRef;

template<typename T>
struct ScalarOrConstRef<T, typename std::enable_if<std::is_scalar<T>::value>::type> {
  using type = T;
};

template<typename T>
struct ScalarOrConstRef<T, typename std::enable_if<!std::is_scalar<T>::value>::type> {
  using type = const T&;
};

}  // namespace detail

template<typename T>
using scalar_or_const_ref_t = typename detail::ScalarOrConstRef<T>::type;

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_TYPE_TRAITS_H_
