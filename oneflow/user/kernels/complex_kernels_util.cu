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
#ifdef WITH_CUDA
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/complex_kernels_util.h"
#include <cuComplex.h>

namespace oneflow {

namespace user_op {

template<typename dtype_x, typename dtype_out>
__global__ void RealCUDA(const dtype_x* x, dtype_out* out, int64_t cnt) {
  CUDA_1D_KERNEL_LOOP(i, cnt) { out[i] = x[i].x; }
}

template<typename dtype_dout, typename dtype_dx>
__global__ void RealGradCUDA(const dtype_dout* dout, dtype_dx* dx, int64_t cnt) {
  CUDA_1D_KERNEL_LOOP(i, cnt) { dx[i] = dtype_dx{dout[i], 0.0}; }
}

template<typename dtype_x, typename dtype_out>
__global__ void ImagCUDA(const dtype_x* x, dtype_out* out, int64_t cnt) {
  CUDA_1D_KERNEL_LOOP(i, cnt) { out[i] = x[i].y; }
}

template<typename dtype_dout, typename dtype_dx>
__global__ void ImagGradCUDA(const dtype_dout* dout, dtype_dx* dx, int64_t cnt) {
  CUDA_1D_KERNEL_LOOP(i, cnt) { dx[i] = dtype_dx{0.0, dout[i]}; }
}

template<typename dtype>
__global__ void ConjPhysicalCUDA(const dtype* x, dtype* out, int64_t cnt) {
  CUDA_1D_KERNEL_LOOP(i, cnt) { out[i] = dtype{x[i].x, -x[i].y}; }
}

template<typename dtype_x, typename dtype_out>
struct RealFunctor<DeviceType::kCUDA, dtype_x, dtype_out> final {
  void operator()(ep::Stream* stream, const dtype_x* x, dtype_out* out, int64_t cnt) {
    RUN_CUDA_KERNEL((RealCUDA<dtype_x, dtype_out>), stream, cnt, x, out, cnt);
  }
};

INSTANTIATE_REAL_FUNCTOR(DeviceType::kCUDA, cuComplex, float)
INSTANTIATE_REAL_FUNCTOR(DeviceType::kCUDA, cuDoubleComplex, double)

template<typename dtype_dout, typename dtype_dx>
struct RealGradFunctor<DeviceType::kCUDA, dtype_dout, dtype_dx> final {
  void operator()(ep::Stream* stream, const dtype_dout* dout, dtype_dx* dx, int64_t cnt) {
    RUN_CUDA_KERNEL((RealGradCUDA<dtype_dout, dtype_dx>), stream, cnt, dout, dx, cnt);
  }
};

INSTANTIATE_REAL_GRAD_FUNCTOR(DeviceType::kCUDA, float, cuComplex)
INSTANTIATE_REAL_GRAD_FUNCTOR(DeviceType::kCUDA, double, cuDoubleComplex)

template<typename dtype_x, typename dtype_out>
struct ImagFunctor<DeviceType::kCUDA, dtype_x, dtype_out> final {
  void operator()(ep::Stream* stream, const dtype_x* x, dtype_out* out, int64_t cnt) {
    RUN_CUDA_KERNEL((ImagCUDA<dtype_x, dtype_out>), stream, cnt, x, out, cnt);
  }
};

INSTANTIATE_IMAG_FUNCTOR(DeviceType::kCUDA, cuComplex, float)
INSTANTIATE_IMAG_FUNCTOR(DeviceType::kCUDA, cuDoubleComplex, double)

template<typename dtype_dout, typename dtype_dx>
struct ImagGradFunctor<DeviceType::kCUDA, dtype_dout, dtype_dx> final {
  void operator()(ep::Stream* stream, const dtype_dout* dout, dtype_dx* dx, int64_t cnt) {
    RUN_CUDA_KERNEL((ImagGradCUDA<dtype_dout, dtype_dx>), stream, cnt, dout, dx, cnt);
  }
};

INSTANTIATE_IMAG_GRAD_FUNCTOR(DeviceType::kCUDA, float, cuComplex)
INSTANTIATE_IMAG_GRAD_FUNCTOR(DeviceType::kCUDA, double, cuDoubleComplex)

template<typename dtype>
struct ConjPhysicalFunctor<DeviceType::kCUDA, dtype> final {
  void operator()(ep::Stream* stream, const dtype* x, dtype* out, int64_t cnt) {
    RUN_CUDA_KERNEL((ConjPhysicalCUDA<dtype>), stream, cnt, x, out, cnt);
  }
};

INSTANTIATE_CONJ_PHYSICAL_FUNCTOR(DeviceType::kCUDA, cuComplex)
INSTANTIATE_CONJ_PHYSICAL_FUNCTOR(DeviceType::kCUDA, cuDoubleComplex)

}  // namespace user_op
}  // namespace oneflow

#endif  // WITH_CUDA
