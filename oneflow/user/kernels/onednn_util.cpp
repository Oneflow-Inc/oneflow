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
#include "oneflow/user/kernels/onednn_util.h"

#ifdef WITH_ONEDNN
namespace oneflow {
template<>
dnnl::memory::data_type CppTypeToOneDnnDtype<int8_t>() {
  return dnnl::memory::data_type::s8;
}

template<>
dnnl::memory::data_type CppTypeToOneDnnDtype<uint8_t>() {
  return dnnl::memory::data_type::u8;
}

template<>
dnnl::memory::data_type CppTypeToOneDnnDtype<int32_t>() {
  return dnnl::memory::data_type::s32;
}

template<>
dnnl::memory::data_type CppTypeToOneDnnDtype<float>() {
  return dnnl::memory::data_type::f32;
}

template<>
dnnl::memory::data_type CppTypeToOneDnnDtype<double>() {
  return dnnl::memory::data_type::undef;
}

}  // namespace oneflow
#endif  // WITH_ONEDNN
