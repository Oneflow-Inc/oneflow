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
#ifndef ONEFLOW_CORE_VECTORIZED_VEC_H_
#define ONEFLOW_CORE_VECTORIZED_VEC_H_
#include <functional>
#include "oneflow/core/common/util.h"
#include "oneflow/core/vectorized/cpu_capbility.h"
#include "oneflow/core/vectorized/vec256/vec256.h"
#include "oneflow/core/vectorized/vec512/vec512.h"
#include "oneflow/core/vectorized/vec_default.h"
#include <stdio.h>

namespace oneflow {

template<typename T, typename Enable = void>
class VecFunc;


void tmp_test();

template<typename T>
class VecFunc<T, typename std::enable_if<std::is_same<T, float>::value
                                         || std::is_same<T, double>::value>::type> {
 public:
  static void init() {
    auto capability = compute_cpu_capability();
    switch (capability) {
      case CPUCapability::DEFAULT:
      {
        std::cout << "DEFAULT : NO SIMD  float" << std::endl;
        fmadd_func = VectorizedDefault<T>::fmadd;
        add_func = VectorizedDefault<T>::add;
        sub_func = VectorizedDefault<T>::sub;
        mul_func = VectorizedDefault<T>::mul;
        div_func = VectorizedDefault<T>::div;
        // test_func = tmp_test;
        // auto v_p = *(test_func.target<void(*)()>());
        // printf("v_p is %p\n", v_p);
        // test_func();
        // std::cout <<"test_func address = " << std::addressof(test_func) << std::endl;
        // printf("VectorizedDefault add address = %p \n", VectorizedDefault<T>::add);
        break;
      }
#ifdef WITH_AVX
      case CPUCapability::AVX2:
      {
        std::cout << "AVX2" << std::endl;
        fmadd_func = VectorizedAvx2<T>::fmadd;
        add_func = VectorizedAvx2<T>::add;
        sub_func = VectorizedAvx2<T>::sub;
        mul_func = VectorizedAvx2<T>::mul;
        div_func = VectorizedAvx2<T>::div;
        break;
      }
      case CPUCapability::AVX512:
      {
        std::cout << "AVX512" << std::endl;
        fmadd_func = VectorizedAvx512<T>::fmadd;
        add_func = VectorizedAvx512<T>::add;
        sub_func = VectorizedAvx512<T>::sub;
        mul_func = VectorizedAvx512<T>::mul;
        div_func = VectorizedAvx512<T>::div;
        break;
      }
#endif
      default: break;
    }
  }

  // begin end x y out alpha
  static std::function<void(size_t, size_t, T*, T*, T*, T)> fmadd_func;
  static std::function<void(size_t, size_t, const T*, const T*, T*)> add_func;
  static std::function<void(size_t, size_t, T*, T*, T*)> sub_func;
  static std::function<void(size_t, size_t, T*, T*, T*)> mul_func;
  static std::function<void(size_t, size_t, T*, T*, T*)> div_func;
};

template<typename T>
class VecFunc<
    T, typename std::enable_if<std::is_same<T, int8_t>::value || std::is_same<T, int32_t>::value
                               || std::is_same<T, int64_t>::value >::type> {
 public:
  static void init() {
    auto capability = compute_cpu_capability();
    switch (capability) {
      case CPUCapability::DEFAULT:
        std::cout << "DEFAULT : NO SIMD  int" << std::endl;
        add_func = VectorizedDefault<T>::add;
        sub_func = VectorizedDefault<T>::sub;
        mul_func = VectorizedDefault<T>::mul;
        div_func = VectorizedDefault<T>::div;
        // test_func = tmp_test;
        // std::cout <<"test_func address = " << std::addressof(test_func) << std::endl;
        // printf("VectorizedDefault add address = %p \n", VectorizedDefault<T>::add);

        // auto vp = *(test_func.target<void(*)()>());
        // printf("vp is %p\n", vp);

        break;
#ifdef WITH_AVX
      case CPUCapability::AVX2:
        std::cout << "AVX2" << std::endl;
        add_func = VectorizedAvx2<T>::add;
        sub_func = VectorizedAvx2<T>::sub;
        mul_func = VectorizedAvx2<T>::mul;
        div_func = VectorizedAvx2<T>::div;
        break;
      case CPUCapability::AVX512:
        std::cout << "AVX512" << std::endl;
        add_func = VectorizedAvx512<T>::add;
        sub_func = VectorizedAvx512<T>::sub;
        mul_func = VectorizedAvx512<T>::mul;
        div_func = VectorizedAvx512<T>::div;
        break;
#endif
      default: break;
    }
  }

  // begin end x y out alpha
  static std::function<void(size_t, size_t, const T*, const T*, T*)> add_func;
  static std::function<void(size_t, size_t, T*, T*, T*)> sub_func;
  static std::function<void(size_t, size_t, T*, T*, T*)> mul_func;
  static std::function<void(size_t, size_t, T*, T*, T*)> div_func;
  static std::function<void(void)> test_func;
};

void vectorized_init();

}  // namespace oneflow

#endif
