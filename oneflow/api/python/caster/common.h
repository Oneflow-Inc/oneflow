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
#include <type_traits>

#include <pybind11/pybind11.h>

namespace pybind11 {
namespace detail {

// The condition follows the pybind11 source code
template<typename T>
using IsSupportedByPybind11WhenInsideSharedPtr =
    std::is_base_of<type_caster_base<T>, type_caster<T>>;

}  // namespace detail
}  // namespace pybind11
