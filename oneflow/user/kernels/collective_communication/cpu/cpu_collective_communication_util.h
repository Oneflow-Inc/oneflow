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
#ifndef ONEFLOW_USER_KERNELS_COLLECTIVE_COMMUNICATION_CPU_CPU_COLLECTIVE_COMMUNICATION_UTIL_H_
#define ONEFLOW_USER_KERNELS_COLLECTIVE_COMMUNICATION_CPU_CPU_COLLECTIVE_COMMUNICATION_UTIL_H_

#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace ccl {

inline int64_t RingDecrease(int64_t n, int64_t size) { return (n - 1 + size) % size; }

inline int64_t RingIncrease(int64_t n, int64_t size) { return (n + 1 + size) % size; }

template<typename T, ReduceType reduce_type>
struct ReduceFunctor;

template<typename T>
struct ReduceFunctor<T, kSum> {
  static void Call(size_t size, T* out, const T* in0, const T* in1) {
    size_t thread_num = Singleton<ThreadPool>::Get()->thread_num();
    BalancedSplitter bs(size, thread_num);
    MultiThreadLoop(thread_num, [&](size_t thread_idx) {
      size_t end = bs.At(thread_idx).end();
      for (size_t i = bs.At(thread_idx).begin(); i < end; ++i) { out[i] = in0[i] + in1[i]; }
    });
  }
};

template<typename T>
struct ReduceFunctor<T, kMax> {
  static void Call(size_t size, T* out, const T* in0, const T* in1) {
    size_t thread_num = Singleton<ThreadPool>::Get()->thread_num();
    BalancedSplitter bs(size, thread_num);
    MultiThreadLoop(thread_num, [&](size_t thread_idx) {
      size_t end = bs.At(thread_idx).end();
      for (size_t i = bs.At(thread_idx).begin(); i < end; ++i) {
        out[i] = std::max(in0[i], in1[i]);
      }
    });
  }
};

}  // namespace ccl

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_COLLECTIVE_COMMUNICATION_CPU_CPU_COLLECTIVE_COMMUNICATION_UTIL_H_
