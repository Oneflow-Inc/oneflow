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

void OFgetrfBatched(ep::Stream* stream, int n, float** dA_array, int ldda, int* ipiv_array,
                    int* info_array, int batchsize) {
  OF_CUBLAS_CHECK(cublasSgetrfBatched(stream->As<ep::CudaStream>()->cublas_handle(), n, dA_array,
                                      ldda, ipiv_array, info_array, batchsize));
}
void OFgetrfBatched(ep::Stream* stream, int n, double** dA_array, int ldda, int* ipiv_array,
                    int* info_array, int batchsize) {
  OF_CUBLAS_CHECK(cublasDgetrfBatched(stream->As<ep::CudaStream>()->cublas_handle(), n, dA_array,
                                      ldda, ipiv_array, info_array, batchsize));
}
void OFgetriBatched(ep::Stream* stream, int n, float** dA_array, int ldda, int* ipiv_array,
                    float** dC_array, int lddc, int* info_array, int batchsize) {
  OF_CUBLAS_CHECK(cublasSgetriBatched(stream->As<ep::CudaStream>()->cublas_handle(), n, dA_array,
                                      ldda, ipiv_array, dC_array, lddc, info_array, batchsize));
}
void OFgetriBatched(ep::Stream* stream, int n, double** dA_array, int ldda, int* ipiv_array,
                    double** dC_array, int lddc, int* info_array, int batchsize) {
  OF_CUBLAS_CHECK(cublasDgetriBatched(stream->As<ep::CudaStream>()->cublas_handle(), n, dA_array,
                                      ldda, ipiv_array, dC_array, lddc, info_array, batchsize));
}

}  // namespace

namespace user_op {

template<typename T>
class CudaInvKernel final : public user_op::OpKernel {
 public:
  CudaInvKernel() = default;
  ~CudaInvKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    auto batch_count = BatchCount(x);
    auto matrix_stride = MatrixStride(x);
    auto matrix_size = x->shape_view().At(x->shape_view().NumAxes() - 2);

    const ShapeView& x_shape = x->shape_view();
    const int64_t instance_num = x_shape.Count(0, x_shape.NumAxes() - 2);
    const int64_t infos_bytes = GetCudaAlignedSize(instance_num * sizeof(int));
    const int64_t ipiv_bytes =
        GetCudaAlignedSize(batch_count * x_shape.At(x_shape.NumAxes() - 2) * sizeof(int));
    const int64_t pptr_bytes = GetCudaAlignedSize(batch_count * sizeof(T*));
    int* infos_getrf_ptr = tmp_buffer->mut_dptr<int>();
    int* infos_getrs_ptr =
        reinterpret_cast<int*>(reinterpret_cast<char*>(infos_getrf_ptr) + infos_bytes);
    int* ipiv_ptr = reinterpret_cast<int*>(reinterpret_cast<char*>(infos_getrs_ptr) + infos_bytes);
    T** x_pptr = reinterpret_cast<T**>(reinterpret_cast<char*>(ipiv_ptr) + ipiv_bytes);
    T** y_pptr = reinterpret_cast<T**>(reinterpret_cast<char*>(x_pptr) + pptr_bytes);
    T* x_copy_ptr = reinterpret_cast<T*>(reinterpret_cast<char*>(y_pptr) + pptr_bytes);
    Memcpy<DeviceType::kCUDA>(ctx->stream(), x_copy_ptr, x->dptr<T>(),
                              x_shape.elem_cnt() * sizeof(T));
    ArangeFunctor<DeviceType::kCUDA, int64_t>()(ctx->stream(),
                                                reinterpret_cast<int64_t>(x_copy_ptr),
                                                static_cast<int64_t>(matrix_stride * sizeof(T)),
                                                batch_count, reinterpret_cast<int64_t*>(x_pptr));
    ArangeFunctor<DeviceType::kCUDA, int64_t>()(ctx->stream(),
                                                reinterpret_cast<int64_t>(y->mut_dptr<T>()),
                                                static_cast<int64_t>(matrix_stride * sizeof(T)),
                                                batch_count, reinterpret_cast<int64_t*>(y_pptr));
    Memset<DeviceType::kCUDA>(ctx->stream(), infos_getrf_ptr, 0, infos_bytes);
    Memset<DeviceType::kCUDA>(ctx->stream(), infos_getrs_ptr, 0, infos_bytes);
    Memset<DeviceType::kCUDA>(ctx->stream(), ipiv_ptr, 0, ipiv_bytes);
    OFgetrfBatched(ctx->stream(), matrix_size, x_pptr, matrix_size, ipiv_ptr, infos_getrf_ptr,
                   batch_count);
    OFgetriBatched(ctx->stream(), matrix_size, x_pptr, matrix_size, ipiv_ptr, y_pptr, matrix_size,
                   infos_getrs_ptr, batch_count);
    std::vector<int> infos_getrf_vec_host(batch_count, 0);
    std::vector<int> infos_getrs_vec_host(batch_count, 0);
    OF_CUDA_CHECK(cudaMemcpyAsync(infos_getrf_vec_host.data(), infos_getrf_ptr,
                                  batch_count * sizeof(int), cudaMemcpyDefault,
                                  ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
    OF_CUDA_CHECK(cudaMemcpyAsync(infos_getrs_vec_host.data(), infos_getrs_ptr,
                                  batch_count * sizeof(int), cudaMemcpyDefault,
                                  ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
    CHECK_JUST(ctx->stream()->Sync());
    FOR_RANGE(int64_t, i, 0, batch_count) {
      if (infos_getrf_vec_host[i] > 0) {
        LOG(FATAL) << "(Batch element " << i << "): The diagonal element "
                   << infos_getrf_vec_host[i]
                   << " is zero, the inversion could not be completed because the input matrix is "
                      "singular.";
      }
    }
    FOR_RANGE(int64_t, i, 0, batch_count) {
      if (infos_getrs_vec_host[i] > 0) {
        LOG(FATAL) << "(Batch element " << i << "): The diagonal element "
                   << infos_getrs_vec_host[i]
                   << " is zero, the inversion could not be completed because the input matrix is "
                      "singular.";
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_INV_KERNEL(dtype)                                                       \
  REGISTER_USER_KERNEL("inv")                                                                 \
      .SetCreateFn<CudaInvKernel<dtype>>()                                                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                        \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value))        \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                     \
        const Shape& x_shape = ctx->InputShape("x", 0);                                       \
        auto batch_size = x_shape.Count(0, x_shape.NumAxes() - 2);                            \
        const int64_t instance_num = x_shape.Count(0, x_shape.NumAxes() - 2);                 \
        const int64_t infos_bytes = GetCudaAlignedSize(instance_num * sizeof(int));           \
        const int64_t ipiv_bytes =                                                            \
            GetCudaAlignedSize(batch_size * x_shape.At(x_shape.NumAxes() - 2) * sizeof(int)); \
        const int64_t pptr_bytes = GetCudaAlignedSize(batch_size * sizeof(dtype*));           \
        const int64_t x_copy_bytes = GetCudaAlignedSize(x_shape.elem_cnt() * sizeof(dtype));  \
        return infos_bytes * 2 + ipiv_bytes + pptr_bytes * 2 + x_copy_bytes;                  \
      });

REGISTER_CUDA_INV_KERNEL(float)
REGISTER_CUDA_INV_KERNEL(double)

}  // namespace user_op
}  // namespace oneflow
