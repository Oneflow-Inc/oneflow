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
#ifndef ONEFLOW_USER_KERNELS_MULTI_REDUCE_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_MULTI_REDUCE_KERNEL_UTIL_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/ep/include/stream.h"

namespace oneflow {

template<typename T>
struct MultiReduceParam {
  const T* data;
  size_t size;
};

template<DeviceType device_type, typename T, typename F>
struct MultiReduceSum {
  void operator()(ep::Stream* stream, F func, const std::vector<MultiReduceParam<T>>& params,
                  T* sum);
};

template<typename T, typename F>
struct MultiReduceSum<DeviceType::kCPU, T, F> {
  void operator()(ep::Stream* stream, F func, const std::vector<MultiReduceParam<T>>& params,
                  T* sum) {
    T sum_v = 0;
    FOR_RANGE(size_t, i, 0, params.size()) {
      const auto& p = params[i];
      FOR_RANGE(size_t, j, 0, p.size) { sum_v += func(p.data[j]); }
    }
    *sum = sum_v;
  }
};

template<typename T>
struct Abs {
  OF_DEVICE_FUNC T operator()(T x) const { return x < 0 ? -x : x; }
};

template<typename T>
struct PowByZero {
  OF_DEVICE_FUNC T operator()(T x) const { return x != T(0) ? T(1) : x; }
};

template<typename T>
struct Square {
  OF_DEVICE_FUNC T operator()(T x) const { return x * x; }
};

template<typename T>
struct AbsPow {
  explicit AbsPow(T base) : base_(base) {}

  OF_DEVICE_FUNC T operator()(T x) {
    T abs_x = x < T(0) ? -x : x;
#if defined(__CUDA_ARCH__)
    return pow(abs_x, base_);
#else
    return std::pow(abs_x, base_);
#endif
  }

 private:
  T base_;
};

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_MULTI_REDUCE_KERNEL_UTIL_H_
