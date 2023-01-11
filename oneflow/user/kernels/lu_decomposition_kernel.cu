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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

#if CUDA_VERSION >= 11000
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

static inline size_t PivotStride(const user_op::Tensor* batched_pivot) {
  const int64_t num_axes = batched_pivot->shape_view().NumAxes();
  return batched_pivot->shape_view().At(num_axes - 1);
}

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

    size_t batch_count = BatchCount(x);
    size_t matrix_stride = MatrixStride(x);
    size_t pivot_stride = PivotStride(x);

    std::unique_ptr<ep::primitive::Memcpy> memcpy_primitive =
        ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(ctx->stream()->device_type(),
                                                                  ep::primitive::MemcpyKind::kDtoD);
    CHECK(memcpy_primitive) << "Can not create Memcpy primitive for device type "
                            << ctx->stream()->device_type();
    memcpy_primitive->Launch(stream, LU_ptr, x_ptr, sizeof(T) * x->shape_view().elem_cnt());

    std::vector<int32_t> batched_info(batch_count, -1);
    int32_t* batched_d_info = nullptr;
    int32_t lwork = -1;
    T* d_work = nullptr;

    OF_CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&batched_d_info), batch_count * sizeof(int32_t)));

    for (size_t batch = 0; batch < batch_count; batch++) {
      OFgetrf_bufferSize(stream, m, m, LU_ptr, m, lwork);
      OF_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_work), sizeof(T) * lwork));
      OFgetrf(stream, m, m, LU_ptr + batch * matrix_stride, lda, d_work,
              pivot_ptr + batch * pivot_stride, batched_d_info + batch);
      OF_CUDA_CHECK(cudaFree(d_work));
    }

    OF_CUDA_CHECK(cudaMemcpyAsync(batched_info.data(), batched_d_info,
                                  batch_count * sizeof(int32_t), cudaMemcpyDeviceToHost,
                                  stream->cuda_stream()));
    for (size_t i = 0; i < batched_info.size(); i++) {
      int32_t info = batched_info[i];
      CHECK(info >= 0) << "LU decomposition: " << -info << "-th parameter of batch " << i
                       << " is wrong";
    }
    OF_CUDA_CHECK(cudaFree(batched_d_info));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_LU_DECOMPOSITION_KERNEL(dtype)                   \
  REGISTER_USER_KERNEL("lu_decomposition")                             \
      .SetCreateFn<LUDecompositionKernel<dtype>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_LU_DECOMPOSITION_KERNEL(float)
REGISTER_CUDA_LU_DECOMPOSITION_KERNEL(double)
#endif

}  // namespace user_op
}  // namespace oneflow
