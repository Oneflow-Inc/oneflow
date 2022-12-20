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
#include <cstddef>
#include <cusolverDn.h>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/ep/include/primitive/memcpy.h"
#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/user/kernels/arange_kernel_util.h"

namespace oneflow {

namespace {

static inline size_t BatchCount(const user_op::Tensor* batched_matrices) {
  size_t result = 1;
  for (size_t i = 0; i < batched_matrices->shape_view().NumAxes() - 2; i++) {
    result *= batched_matrices->shape_view().At(i);
  }
  return result;
}

static inline size_t MatrixStride(const user_op::Tensor* batched_matrices) {
  const int64_t num_axes = batched_matrices->shape_view().NumAxes();
  return batched_matrices->shape_view().At(num_axes - 2)
         * batched_matrices->shape_view().At(num_axes - 1);
}
// OF_CUSOLVER_CHECK(
//     cusolverDnSgetrf_bufferSize(stream->cusolver_dn_handle(), m, m, LU_ptr, m, &lwork));

void OFgetrf_bufferSize(ep::Stream* stream, int32_t m, int32_t n, float* dA_array, int32_t lda,
                        int32_t& lwork) {
  OF_CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(stream->As<ep::CudaStream>()->cusolver_dn_handle(),
                                                m, n, dA_array, m, &lwork));
}

void OFgetrf_bufferSize(ep::Stream* stream, int32_t m, int32_t n, double* dA_array, int32_t lda,
                        int32_t& lwork) {
  OF_CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(stream->As<ep::CudaStream>()->cusolver_dn_handle(),
                                                m, n, dA_array, m, &lwork));
}

void OFgetrf(ep::Stream* stream, int32_t m, int32_t n, float* dA_array, int32_t lda, float* d_work,
             int32_t* pivot_ptr, int32_t* d_info) {
  OF_CUSOLVER_CHECK(cusolverDnSgetrf(stream->As<ep::CudaStream>()->cusolver_dn_handle(), m, m,
                                     dA_array, lda, d_work, pivot_ptr, d_info));
}

void OFgetrf(ep::Stream* stream, int32_t m, int32_t n, double* dA_array, int32_t lda,
             double* d_work, int32_t* pivot_ptr, int32_t* d_info) {
  OF_CUSOLVER_CHECK(cusolverDnDgetrf(stream->As<ep::CudaStream>()->cusolver_dn_handle(), m, m,
                                     dA_array, lda, d_work, pivot_ptr, d_info));
}
}  // namespace

// void OFgetrfBatched(ep::Stream* stream, int n, float** dA_array, int ldda, int* ipiv_array,
//                     int* info_array, int batchsize) {
//   OF_CUBLAS_CHECK(cublasSgetrfBatched(stream->As<ep::CudaStream>()->cublas_handle(), n, dA_array,
//                                       ldda, ipiv_array, info_array, batchsize));
// }
// void OFgetrfBatched(ep::Stream* stream, int n, double** dA_array, int ldda, int* ipiv_array,
//                     int* info_array, int batchsize) {
//   OF_CUBLAS_CHECK(cublasDgetrfBatched(stream->As<ep::CudaStream>()->cublas_handle(), n, dA_array,
//                                       ldda, ipiv_array, info_array, batchsize));
// }
// void OFgetriBatched(ep::Stream* stream, int n, float** dA_array, int ldda, int* ipiv_array,
//                     float** dC_array, int lddc, int* info_array, int batchsize) {
//   OF_CUBLAS_CHECK(cublasSgetriBatched(stream->As<ep::CudaStream>()->cublas_handle(), n, dA_array,
//                                       ldda, ipiv_array, dC_array, lddc, info_array, batchsize));
// }
// void OFgetriBatched(ep::Stream* stream, int n, double** dA_array, int ldda, int* ipiv_array,
//                     double** dC_array, int lddc, int* info_array, int batchsize) {
//   OF_CUBLAS_CHECK(cublasDgetriBatched(stream->As<ep::CudaStream>()->cublas_handle(), n, dA_array,
//                                       ldda, ipiv_array, dC_array, lddc, info_array, batchsize));
// }

namespace user_op {

template<typename T>
class LUDecompositionKernel final : public user_op::OpKernel {
 public:
  LUDecompositionKernel() = default;
  ~LUDecompositionKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* LU = ctx->Tensor4ArgNameAndIndex("LU", 0);
    user_op::Tensor* pivot = ctx->Tensor4ArgNameAndIndex("pivot", 0);
    auto stream = ctx->stream()->As<ep::CudaStream>();

    // infer tmp buffer
    const int32_t m = x->shape_view().At(x->shape_view().NumAxes() - 2);
    const int32_t lda = m;
    const T* x_ptr = x->dptr<T>();
    T* LU_ptr = LU->mut_dptr<T>();
    int32_t* pivot_ptr = pivot->mut_dptr<int32_t>();

    std::unique_ptr<ep::primitive::Memcpy> memcpy_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(ctx->stream()->device_type(),
                                                                  ep::primitive::MemcpyKind::kDtoD);
    CHECK(memcpy_primitive) << "Can not create Memcpy primitive for device type "
                            << ctx->stream()->device_type();
    memcpy_primitive->Launch(stream, LU_ptr, x_ptr, sizeof(T) * x->shape_view().elem_cnt());

    int info = 0;

    int32_t* d_info = nullptr; /* error info */
    int lwork = -1;            /* size of workspace */
    T* d_work = nullptr;       /* device workspace for getrf */

    OF_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int32_t)));
    OFgetrf_bufferSize(stream, m, m, LU_ptr, m, lwork);
    OF_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_work), sizeof(T) * lwork));
    OFgetrf(stream, m, m, LU_ptr, lda, d_work, pivot_ptr, d_info);
    OF_CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int32_t), cudaMemcpyDeviceToHost,
                                  stream->cuda_stream()));
    CHECK(info >= 0) << "LU decomposition: " << -info << "-th parameter is wrong";

    OF_CUDA_CHECK(cudaFree(d_info));
    OF_CUDA_CHECK(cudaFree(d_work));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_LU_DECOMPOSITION_KERNEL(dtype)                   \
  REGISTER_USER_KERNEL("lu_decomposition")                             \
      .SetCreateFn<LUDecompositionKernel<dtype>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_LU_DECOMPOSITION_KERNEL(float)

}  // namespace user_op
}  // namespace oneflow
