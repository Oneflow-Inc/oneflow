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
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

Shape CreatePreluLeftExtendedShape(const ShapeView& shape) {
  DimVector dim_vec(shape.NumAxes());
  dim_vec.at(0) = 1LL;
  dim_vec.at(1) = shape.At(1);
  for (int i = 2; i < shape.NumAxes(); i++) { dim_vec.at(i) = 1LL; }
  return Shape(std::move(dim_vec));
}

template<typename T>
__global__ void BroadcastPReluSingleAlphaForwardGpu(const int32_t elem_cnt,
                                                    const int32_t alpha_size,
                                                    const int32_t inner_size, const T* x,
                                                    const T* alpha, T* y) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T x_i = x[i];
    y[i] = x_i > 0 ? x_i : x_i * alpha[0];
  }
}

template<typename T>
__global__ void BroadcastPReluSingleAlphaBackwardGpu(const int32_t elem_cnt,
                                                     const int32_t alpha_size,
                                                     const int32_t inner_size, const T* x,
                                                     const T* alpha, const T* dy, T* dx,
                                                     T* alpha_diff) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T x_i = x[i];
    const T dy_i = dy[i];
    T dx_i = 0;
    T alpha_diff_i = 0;
    if (x_i > 0) {
      dx_i = dy_i;
      alpha_diff_i = 0;
    } else {
      dx_i = dy_i * alpha[0];
      alpha_diff_i = dy_i * x_i;
    }
    dx[i] = dx_i;
    alpha_diff[i] = alpha_diff_i;
  }
}

template<>
__global__ void BroadcastPReluSingleAlphaForwardGpu<half>(const int32_t elem_cnt,
                                                          const int32_t alpha_size,
                                                          const int32_t inner_size, const half* x,
                                                          const half* alpha, half* y) {
  half zero = static_cast<half>(0.0);
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const half x_i = x[i];
    y[i] = x_i > zero ? x_i : __hmul(x_i, alpha[0]);
  }
}

template<>
__global__ void BroadcastPReluSingleAlphaBackwardGpu<half>(const int32_t elem_cnt,
                                                           const int32_t alpha_size,
                                                           const int32_t inner_size, const half* x,
                                                           const half* alpha, const half* dy,
                                                           half* dx, half* alpha_diff) {
  half zero = static_cast<half>(0.0);
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const half x_i = x[i];
    const half dy_i = dy[i];
    half dx_i = 0;
    half alpha_diff_i = 0;
    if (x_i > zero) {
      dx_i = dy_i;
      alpha_diff_i = 0;
    } else {
      dx_i = __hmul(dy_i, alpha[0]);
      alpha_diff_i = __hmul(dy_i, x_i);
    }
    dx[i] = dx_i;
    alpha_diff[i] = alpha_diff_i;
  }
}

// template<typename T>
// __global__ void BroadcastPReluMultiAlphaNaiveForwardGpu(const int32_t elem_cnt, const int32_t alpha_size,
//                                                    const int32_t inner_size, const T* x,
//                                                    const T* alpha, T* y) {
//   CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
//     const T x_i = x[i];
//     int32_t i_div_inner_size = i / inner_size;
//     int32_t idx_sub_alpha = i_div_inner_size / alpha_size * alpha_size;
//     int32_t alpha_i = i_div_inner_size - idx_sub_alpha;
//     y[i] = x_i > 0 ? x_i : x_i * alpha[alpha_i];
//   }
// }

template<typename T>
__global__ void BroadcastPReluMultiAlphaNaiveForwardGpu(const int32_t elem_cnt, const int32_t alpha_size,
                                                   const int32_t inner_size, const T* x,
                                                   const T* alpha, T* y) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T x_i = x[i];
    // int32_t i_div_inner_size = i / inner_size;
    // int32_t idx_sub_alpha = i_div_inner_size / alpha_size * alpha_size;
    // int32_t alpha_i = i_div_inner_size - idx_sub_alpha;
    int32_t alpha_idx = (i / inner_size) % alpha_size; 
    y[i] = x_i > 0 ? x_i : x_i * alpha[alpha_idx];
  }
}


template<typename T, typename IndexType, int pack_size>
__global__ void PReluForwardMultiAlphaGpu(const IndexType elem_cnt, const IndexType alpha_size,
                                          const IndexType inner_size, const IndexType mul_size,
                                          const T* x, const T* alpha, T* y) {
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  using LoadType = cuda::elementwise::PackType<T, pack_size>;
  using LoadPack = cuda::elementwise::Pack<T, pack_size>;
  T zero_val = static_cast<T>(0);
  for (int64_t linear_index = global_thread_id * pack_size; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x * pack_size) {
    IndexType idx_sub_alpha = linear_index / mul_size * alpha_size;
    IndexType alpha_idx = linear_index / inner_size - idx_sub_alpha;

    const LoadType* x_load = reinterpret_cast<const LoadType*>(x + linear_index);
    LoadPack x_vec;
    x_vec.storage = *x_load;
    
    LoadPack y_vec;

    T alpha_val = alpha[alpha_idx];
#pragma unroll
    for (int i = 0; i < pack_size; i++) {
      y_vec.elem[i] = x_vec.elem[i] > zero_val ? x_vec.elem[i] : x_vec.elem[i] * alpha_val;
    }
    *(reinterpret_cast<LoadType*>(y + linear_index)) = y_vec.storage;
  }
}


template<typename T>
__global__ void BroadcastPReluMultiAlphaBackwardGpu(const int32_t elem_cnt,
                                                    const int32_t alpha_size,
                                                    const int32_t inner_size, const T* x,
                                                    const T* alpha, const T* dy, T* dx,
                                                    T* alpha_diff) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T x_i = x[i];
    const T dy_i = dy[i];
    int32_t i_div_inner_size = i / inner_size;
    int32_t idx_sub_alpha = i_div_inner_size / alpha_size * alpha_size;
    int32_t alpha_i = i_div_inner_size - idx_sub_alpha;
    T dx_i = 0;
    T alpha_diff_i = 0;
    if (x_i > 0) {
      dx_i = dy_i;
      alpha_diff_i = 0;
    } else {
      dx_i = dy_i * alpha[alpha_i];
      alpha_diff_i = dy_i * x_i;
    }
    dx[i] = dx_i;
    alpha_diff[i] = alpha_diff_i;
  }
}

template<>
__global__ void BroadcastPReluMultiAlphaNaiveForwardGpu<half>(const int32_t elem_cnt,
                                                         const int32_t alpha_size,
                                                         const int32_t inner_size, const half* x,
                                                         const half* alpha, half* y) {
  half zero = static_cast<half>(0.0);
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const half x_i = x[i];
    int32_t i_div_inner_size = i / inner_size;
    int32_t idx_sub_alpha = i_div_inner_size / alpha_size * alpha_size;
    int32_t alpha_i = i_div_inner_size - idx_sub_alpha;
    y[i] = x_i > zero ? x_i : __hmul(x_i, alpha[alpha_i]);
  }
}

template<>
__global__ void BroadcastPReluMultiAlphaBackwardGpu<half>(const int32_t elem_cnt,
                                                          const int32_t alpha_size,
                                                          const int32_t inner_size, const half* x,
                                                          const half* alpha, const half* dy,
                                                          half* dx, half* alpha_diff) {
  half zero = static_cast<half>(0.0);
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const half x_i = x[i];
    const half dy_i = dy[i];
    int32_t i_div_inner_size = i / inner_size;
    int32_t idx_sub_alpha = i_div_inner_size / alpha_size * alpha_size;
    int32_t alpha_i = i_div_inner_size - idx_sub_alpha;
    half dx_i = 0;
    half alpha_diff_i = 0;
    if (x_i > zero) {
      dx_i = dy_i;
      alpha_diff_i = 0;
    } else {
      dx_i = __hmul(dy_i, alpha[alpha_i]);
      alpha_diff_i = __hmul(dy_i, x_i);
    }
    dx[i] = dx_i;
    alpha_diff[i] = alpha_diff_i;
  }
}

constexpr int32_t kBlockSize = 256; 

template<typename T, typename IndexType, int32_t pack_size>
void DispatchPreluForwardPackSize(ep::Stream* stream, const int64_t elem_cnt, const int64_t alpha_size,
  const int64_t inner_size, const T* x, const T* alpha, T* y){
  const int64_t pack_num = elem_cnt / pack_size;
  int grid_size;
  cudaError_t err = cuda::elementwise::GetNumBlocks(pack_num, &grid_size);
  const int64_t alpha_inner_size = alpha_size * inner_size;
    
  if(pack_size>=8 && inner_size % 8 == 0 ){
    printf("Here use packsize is 8\n");
    PReluForwardMultiAlphaGpu<T, IndexType, 8>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            elem_cnt, alpha_size, inner_size, alpha_inner_size, x, alpha, y);
  }else if(pack_size>=4 && inner_size%4 == 0 ){
    printf("Here use packsize is 4\n");

    PReluForwardMultiAlphaGpu<T, IndexType, 4>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            elem_cnt, alpha_size, inner_size, alpha_inner_size, x, alpha, y);
  }else if(pack_size>=2 && inner_size%2 == 0 ){
    printf("Here use packsize is 2\n");

    PReluForwardMultiAlphaGpu<T, IndexType, 2>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            elem_cnt, alpha_size, inner_size, alpha_inner_size, x, alpha, y);
  }else{
    printf("Here use Naive prelu alpha Kernel!\n");

    BroadcastPReluMultiAlphaNaiveForwardGpu<T>
      <<<grid_size, kBlockSize, 0,
          stream->As<ep::CudaStream>()->cuda_stream()>>>(
          elem_cnt, alpha_size, inner_size, x, alpha, y);
  }
}


template<typename T>
void DispatchPreluForwardIndex(ep::Stream* stream, const int64_t elem_cnt, const int64_t alpha_size,
                               const int64_t inner_size, const T* x, const T* alpha, T* y) {
  // constexpr int max_pack_size = cuda::elementwise::PackSize<T>();
  // int32_t pack_size = max_pack_size; 

  // while(inner_size % pack_size !=0){
  //   pack_size = pack_size / 2; 
  // }

  // printf("Here actual Packsize is %d \n", pack_size); 

  // const int64_t pack_num = elem_cnt / pack_size;
  // int grid_size;
  // cudaError_t err = cuda::elementwise::GetNumBlocks(pack_num, &grid_size);
  // const int64_t alpha_inner_size = alpha_size * inner_size;
  constexpr int pack_size = cuda::elementwise::PackSize<T>();

  if (elem_cnt < GetMaxVal<int32_t>()) {
    DispatchPreluForwardPackSize<T, int32_t, pack_size>(stream, elem_cnt, alpha_size, inner_size, x, alpha, y);
  } else {
    DispatchPreluForwardPackSize<T, int64_t, pack_size>(stream, elem_cnt, alpha_size, inner_size, x, alpha, y);
  }
}

}  // namespace

template<typename T>
class GpuPReluKernel final : public user_op::OpKernel {
 public:
  GpuPReluKernel() = default;
  ~GpuPReluKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* alpha = ctx->Tensor4ArgNameAndIndex("alpha", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    const int32_t batch = x->shape().At(0);
    const int32_t channels = x->shape().At(1);
    const int32_t alpha_size = alpha->shape().elem_cnt();
    const int32_t inner_size = elem_cnt / batch / channels;

    if (alpha_size == 1) {
      BroadcastPReluSingleAlphaForwardGpu<T>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
             ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              elem_cnt, alpha_size, inner_size, x->dptr<T>(), alpha->dptr<T>(), y->mut_dptr<T>());
    } else {
      // BroadcastPReluMultiAlphaNaiveForwardGpu<T>
      //     <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
      //        ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
      //         elem_cnt, alpha_size, inner_size, x->dptr<T>(), alpha->dptr<T>(), y->mut_dptr<T>());
      DispatchPreluForwardIndex<T>(
        ctx->stream(), elem_cnt, alpha_size, inner_size, reinterpret_cast<const T*>(x->dptr()),
        reinterpret_cast<const T*>(alpha->dptr()), reinterpret_cast<T*>(y->mut_dptr()));
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_PRELU_KERNEL(dtype)                                             \
  REGISTER_USER_KERNEL("prelu").SetCreateFn<GpuPReluKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCUDA)                                 \
      && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_PRELU_KERNEL(half)
REGISTER_CUDA_PRELU_KERNEL(float)
REGISTER_CUDA_PRELU_KERNEL(double)

template<typename T>
class GpuPReluGradKernel final : public user_op::OpKernel {
 public:
  GpuPReluGradKernel() = default;
  ~GpuPReluGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* alpha = ctx->Tensor4ArgNameAndIndex("alpha", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    user_op::Tensor* alpha_diff = ctx->Tensor4ArgNameAndIndex("alpha_diff", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    T* broadcasted_alpha_diff = tmp_buffer->mut_dptr<T>();
    T* reduce_sum_tmp_buf = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>()
                                                 + GetCudaAlignedSize(elem_cnt * sizeof(T)));

    const Shape& left_extended_shape = CreatePreluLeftExtendedShape(ShapeView(x->shape()));

    const int32_t batch = x->shape().At(0);
    const int32_t channels = x->shape().At(1);
    const int32_t alpha_size = alpha->shape().elem_cnt();
    const int32_t inner_size = elem_cnt / batch / channels;
    if (alpha_size == 1) {
      BroadcastPReluSingleAlphaBackwardGpu<T>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
             ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              elem_cnt, alpha_size, inner_size, x->dptr<T>(), alpha->dptr<T>(), dy->dptr<T>(),
              dx->mut_dptr<T>(), broadcasted_alpha_diff);
    } else {
      BroadcastPReluMultiAlphaBackwardGpu<T>
          <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
             ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              elem_cnt, alpha_size, inner_size, x->dptr<T>(), alpha->dptr<T>(), dy->dptr<T>(),
              dx->mut_dptr<T>(), broadcasted_alpha_diff);
    }

    NdarrayUtil<DeviceType::kCUDA, T>::ReduceSum(
        ctx->stream(), XpuVarNdarray<T>(left_extended_shape, alpha_diff->mut_dptr<T>()),
        XpuVarNdarray<const T>(x->shape(), broadcasted_alpha_diff),
        XpuVarNdarray<T>(x->shape(), reduce_sum_tmp_buf));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_PRELU_GRAD_KERNEL(dtype)                                          \
  REGISTER_USER_KERNEL("prelu_grad")                                                    \
      .SetCreateFn<GpuPReluGradKernel<dtype>>()                                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                               \
        const Shape& in_shape = ctx->InputShape("x", 0);                                \
        const Shape& alpha_shape = ctx->InputShape("alpha", 0);                         \
        const int64_t tmp_buffer_size =                                                 \
            2 * GetCudaAlignedSize(in_shape.elem_cnt() * sizeof(dtype));                \
        return tmp_buffer_size;                                                         \
      });

REGISTER_CUDA_PRELU_GRAD_KERNEL(half)
REGISTER_CUDA_PRELU_GRAD_KERNEL(float)
REGISTER_CUDA_PRELU_GRAD_KERNEL(double)

}  // namespace oneflow
