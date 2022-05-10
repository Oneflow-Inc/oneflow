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
#ifndef ONEFLOW_CORE_KERNEL_UTIL_CUDA_BLAS_INTERFACE_H_
#define ONEFLOW_CORE_KERNEL_UTIL_CUDA_BLAS_INTERFACE_H_

#include "oneflow/core/kernel/util/blas_interface.h"

namespace oneflow {

class Blob;

template<>
struct BlasIf<DeviceType::kCUDA> {
  static void OFGemm(ep::Stream* stream, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                     const int m, const int n, const int k, const double alpha, const float* a,
                     const float* b, const double beta, float* c);
  static void OFGemm(ep::Stream* stream, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                     const int m, const int n, const int k, const double alpha, const double* a,
                     const double* b, const double beta, double* c);
  static void OFGemm(ep::Stream* stream, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                     const int m, const int n, const int k, const double alpha, const float16* a,
                     const float16* b, const double beta, float16* c);
  static void OFgetrfBatched(ep::Stream* stream, int n, float** dA_array, int ldda, int* ipiv_array,
                             int* info_array, int batchsize);
  static void OFgetrfBatched(ep::Stream* stream, int n, double** dA_array, int ldda,
                             int* ipiv_array, int* info_array, int batchsize);
  static void OFgetriBatched(ep::Stream* stream, int n, float** dA_array, int ldda, int* ipiv_array,
                             float** dC_array, int lddc, int* info_array, int batchsize);
  static void OFgetriBatched(ep::Stream* stream, int n, double** dA_array, int ldda,
                             int* ipiv_array, double** dC_array, int lddc, int* info_array,
                             int batchsize);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_UTIL_CUDA_BLAS_INTERFACE_H_
