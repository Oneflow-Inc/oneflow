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

#ifndef ONEFLOW_USER_KERNELS_CUFFT_PLAN_CACHE_H_
#define ONEFLOW_USER_KERNELS_CUFFT_PLAN_CACHE_H_

#include <cufft.h>
#include <cufftXt.h>
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

constexpr int max_rank = 3;

}

struct CuFFtParams {
  int32_t ndim;
  int32_t output_shape[max_rank + 1];
  int32_t input_shape[max_rank + 1];
  int32_t input_strides[max_rank + 1];
  int32_t output_strides[max_rank + 1];
  int32_t* rank;
  int32_t batch;
  CuFFtParams(int32_t dims, int32_t* r, const Stride& in_strides,  // NOLINT
              const Stride& out_strides, const Shape& in_shape, const Shape& out_shape, int32_t b)
      : ndim(dims), rank(r), batch(b) {
    std::copy(in_strides.begin(), in_strides.end(), input_strides);
    std::copy(out_strides.begin(), out_strides.end(), output_strides);
    std::copy(in_shape.begin(), in_shape.end(), input_shape);
    std::copy(out_shape.begin(), out_shape.end(), output_shape);
  }
};

template<typename T, typename C>
class CuFFtConfig {
 public:
  CuFFtConfig(const CuFFtConfig&) = delete;
  CuFFtConfig& operator=(CuFFtConfig const&) = delete;
  ~CuFFtConfig() = default;

  explicit CuFFtConfig(CuFFtParams& params) {  // NOLINT
    infer_cufft_type_();
    cufftPlanMany(&plan_handle_, params.ndim, params.rank, params.input_shape,
                  params.input_strides[0], params.input_strides[1], params.output_shape,
                  params.output_strides[0], params.output_strides[1], exectype_, params.batch);
  }

  void excute_plan(const T* in, C* out) {
    switch (exectype_) {
      case CUFFT_R2C: cufftExecR2C(plan_handle_, (cufftReal*)in, (cufftComplex*)out); break;

      case CUFFT_D2Z:
        cufftExecD2Z(plan_handle_, (cufftDoubleReal*)in, (cufftDoubleComplex*)out);
        break;
      default: break;
    }
  }

 private:
  // infer representing the FFT type(暂时只支持R2C,D2Z)
  void infer_cufft_type_() {
    bool isDouble = std::is_same<double, T>::value;
    if (isDouble) {
      exectype_ = CUFFT_D2Z;
    } else {
      exectype_ = CUFFT_R2C;
    }
  }

  cufftHandle plan_handle_;
  cufftType exectype_;
};

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_CUFFT_PLAN_CACHE_H_
