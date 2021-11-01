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
#include <cstdint>
#include <iostream>
#include "vec.h"

namespace oneflow {

void vectorized_init() {
  VecFunc<float>::init();
  VecFunc<double>::init();
  VecFunc<int8_t>::init();
  VecFunc<int32_t>::init();
  VecFunc<int64_t>::init();
}

// COMMAND(vectorized_init());

template<typename T>
std::function<void(size_t, size_t, const T*, const T*, T*, const T)>
    VecFunc<T, typename std::enable_if<std::is_same<T, float>::value
                                       || std::is_same<T, double>::value>::type>::fmadd_func;
template<typename T>
std::function<void(size_t, size_t, const T*, const T*, T*)>
    VecFunc<T, typename std::enable_if<std::is_same<T, float>::value
                                       || std::is_same<T, double>::value>::type>::add_func;
template<typename T>
std::function<void(size_t, size_t, const T*, const T*, T*)> VecFunc<
    T, typename std::enable_if<std::is_same<T, int8_t>::value || std::is_same<T, int32_t>::value
                               || std::is_same<T, int64_t>::value>::type>::add_func;

template<typename T>
std::function<void(size_t, size_t, const T*, const T*, T*)>
    VecFunc<T, typename std::enable_if<std::is_same<T, float>::value
                                       || std::is_same<T, double>::value>::type>::sub_func;
template<typename T>
std::function<void(size_t, size_t, const T*, const T*, T*)> VecFunc<
    T, typename std::enable_if<std::is_same<T, int8_t>::value || std::is_same<T, int32_t>::value
                               || std::is_same<T, int64_t>::value>::type>::sub_func;

template<typename T>
std::function<void(size_t, size_t, const T*, const T*, T*)>
    VecFunc<T, typename std::enable_if<std::is_same<T, float>::value
                                       || std::is_same<T, double>::value>::type>::mul_func;
template<typename T>
std::function<void(size_t, size_t, const T*, const T*, T*)> VecFunc<
    T, typename std::enable_if<std::is_same<T, int8_t>::value || std::is_same<T, int32_t>::value
                               || std::is_same<T, int64_t>::value>::type>::mul_func;

template<typename T>
std::function<void(size_t, size_t, const T*, const T*, T*)>
    VecFunc<T, typename std::enable_if<std::is_same<T, float>::value
                                       || std::is_same<T, double>::value>::type>::div_func;
template<typename T>
std::function<void(size_t, size_t, const T*, const T*, T*)> VecFunc<
    T, typename std::enable_if<std::is_same<T, int8_t>::value || std::is_same<T, int32_t>::value
                               || std::is_same<T, int64_t>::value>::type>::div_func;

}  // namespace oneflow