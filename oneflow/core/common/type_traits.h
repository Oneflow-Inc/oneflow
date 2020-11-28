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
