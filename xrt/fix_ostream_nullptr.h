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

#ifndef ONEFLOW_XRT_FIX_OSTREAM_NULLPTR_H_
#define ONEFLOW_XRT_FIX_OSTREAM_NULLPTR_H_

#include <iostream>

// This header fix the overload error below, while using glog instead of
// tensorflow/core/platform/default/logging.h: In file included from
// oneflow/oneflow/core/common/util.h:22,
//                  from oneflow/oneflow/xrt/xla/xla_allocator.h:19,
//                  from oneflow/oneflow/xrt/xla/xla_resource_manager.h:25,
//                  from oneflow/oneflow/xrt/xla/xla_resource_manager.cpp:18:
// oneflow/build/third_party_install/glog/install/include/glog/logging.h: In instantiation of ‘void
// google::MakeCheckOpValueString(std::ostream*, const T&) [with T = std::nullptr_t; std::ostream =
// std::basic_ostream<char>]’:
// oneflow/build/third_party_install/glog/install/include/glog/logging.h:693:25:   required from
// ‘std::string* google::MakeCheckOpString(const T1&, const T2&, const char*) [with T1 =
// std::nullptr_t; T2 = xla::HloComputation*; std::string = std::__cxx11::basic_string<char>]’
// oneflow/build/third_party_install/glog/install/include/glog/logging.h:718:1:   required from
// ‘std::string* google::Check_NEImpl(const T1&, const T2&, const char*) [with T1 = std::nullptr_t;
// T2 = xla::HloComputation*; std::string = std::__cxx11::basic_string<char>]’
// oneflow/build/third_party_install/tensorflow/include/tensorflow_inc/tensorflow/compiler/xla/service/hlo_module.h:116:5:
// required from here oneflow/build/third_party_install/glog/install/include/glog/logging.h:638:9:
// error: ambiguous overload for ‘operator<<’ (operand types are ‘std::ostream’ {aka
// ‘std::basic_ostream<char>’} and ‘std::nullptr_t’)
//   638 |   (*os) << v;
//       |   ~~~~~~^~~~
namespace std {
inline ostream& operator<<(ostream& os, nullptr_t) { return os << "nullptr"; }
}  // namespace std

#endif  // ONEFLOW_XRT_FIX_OSTREAM_NULLPTR_H_
