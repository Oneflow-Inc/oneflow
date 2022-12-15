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

Shape CreatePreluLeftExtendedShape(const ShapeView& shape, const int32_t alpha_size) {
  DimVector dim_vec(shape.NumAxes());
  dim_vec.at(0) = 1LL;
  dim_vec.at(1) = alpha_size;
  for (int i = 2; i < shape.NumAxes(); i++) { dim_vec.at(i) = 1LL; }
  return Shape(std::move(dim_vec));
}

template<typename T>
struct PreluForwardSingleAlphaFunctor {
  OF_DEVICE_FUNC explicit PreluForwardSingleAlphaFunctor(const T alpha) : alpha(alpha) {}
  __device__ T operator()(T x) const { return (x > static_cast<T>(0.0)) ? x : (alpha * x); }
  const T alpha;
};

template<typename T>
struct PreluForwardSingleAlphaPtrFunctor {
  OF_DEVICE_FUNC explicit PreluForwardSingleAlphaPtrFunctor(const T* alpha_ptr)
      : alpha_ptr(alpha_ptr) {}
  __device__ PreluForwardSingleAlphaFunctor<T> operator()() const {
    return PreluForwardSingleAlphaFunctor<T>(*alpha_ptr);
  }
  const T* alpha_ptr;
};

template<typename T, typename IndexType, int pack_size, bool tail, bool alpha_requires_grad>
__global__ void PReluBackwardSingleAlphaGpu(const IndexType elem_cnt, const int64_t n_tail,
                                            const T* x, const T* alpha, const T* dy, T* dx,
                                            T* alpha_diff, const T* tail_x, const T* tail_dy,
                                            T* tail_dx, T* tail_alpha_diff) {
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  using LoadType = cuda::elementwise::PackType<T, pack_size>;
  using LoadPack = cuda::elementwise::Pack<T, pack_size>;
  T zero_val = static_cast<T>(0);
  T alpha_val = alpha[0];

  for (int64_t linear_index = global_thread_id * pack_size; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x * pack_size) {
    const LoadType* x_load = reinterpret_cast<const LoadType*>(x + linear_index);
    LoadPack x_vec;
    x_vec.storage = *x_load;

    const LoadType* dy_load = reinterpret_cast<const LoadType*>(dy + linear_index);
    LoadPack dy_vec;
    dy_vec.storage = *dy_load;

    LoadPack dx_vec;
    T zero_val = static_cast<T>(0.0);
    if (alpha_requires_grad) {
      LoadPack dalpha_vec;
#pragma unroll
      for (int i = 0; i < pack_size; i++) {
        if (x_vec.elem[i] > zero_val) {
          dx_vec.elem[i] = dy_vec.elem[i];
          dalpha_vec.elem[i] = zero_val;
        } else {
          dx_vec.elem[i] = dy_vec.elem[i] * alpha_val;
          dalpha_vec.elem[i] = dy_vec.elem[i] * x_vec.elem[i];
        }
      }
      *(reinterpret_cast<LoadType*>(dx + linear_index)) = dx_vec.storage;
      *(reinterpret_cast<LoadType*>(alpha_diff + linear_index)) = dalpha_vec.storage;
    } else {
#pragma unroll
      for (int i = 0; i < pack_size; i++) {
        if (x_vec.elem[i] > zero_val) {
          dx_vec.elem[i] = dy_vec.elem[i];
        } else {
          dx_vec.elem[i] = dy_vec.elem[i] * alpha_val;
        }
      }
      *(reinterpret_cast<LoadType*>(dx + linear_index)) = dx_vec.storage;
    }
  }

  if (tail && global_thread_id < n_tail) {
    const T tail_dy_val = tail_dy[global_thread_id];
    if (tail_x[global_thread_id] > zero_val) {
      tail_dx[global_thread_id] = tail_dy_val;
      if (alpha_requires_grad) { tail_alpha_diff[global_thread_id] = zero_val; }
    } else {
      tail_dx[global_thread_id] = alpha_val * tail_dy_val;
      if (alpha_requires_grad) {
        tail_alpha_diff[global_thread_id] = tail_x[global_thread_id] * tail_dy_val;
      }
    }
  }
}

template<typename T>
__global__ void BroadcastPReluMultiAlphaNaiveForwardGpu(const int32_t elem_cnt,
                                                        const int32_t alpha_size,
                                                        const int32_t inner_size, const T* x,
                                                        const T* alpha, T* y) {
  const T zero_val = static_cast<T>(0.0);
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T x_i = x[i];
    int32_t alpha_idx = (i / inner_size) % alpha_size;
    y[i] = x_i > zero_val ? x_i : x_i * alpha[alpha_idx];
  }
}

template<typename T, typename IndexType, int pack_size>
__global__ void PReluForwardMultiAlphaGpu(const IndexType elem_cnt, const IndexType alpha_size,
                                          const IndexType inner_size, const T* x, const T* alpha,
                                          T* y) {
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  using LoadType = cuda::elementwise::PackType<T, pack_size>;
  using LoadPack = cuda::elementwise::Pack<T, pack_size>;
  T zero_val = static_cast<T>(0);
  for (int64_t linear_index = global_thread_id * pack_size; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x * pack_size) {
    IndexType alpha_idx = (linear_index / inner_size) % alpha_size;

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

template<typename T, bool alpha_requires_grad>
__global__ void BroadcastPReluMultiAlphaNaiveBackwardGpu(const int32_t elem_cnt,
                                                         const int32_t alpha_size,
                                                         const int32_t inner_size, const T* x,
                                                         const T* alpha, const T* dy, T* dx,
                                                         T* alpha_diff) {
  const T zero_val = static_cast<T>(0.0);
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T x_i = x[i];
    const T dy_i = dy[i];
    int32_t alpha_i = (i / inner_size) % alpha_size;
    if (x_i > zero_val) {
      dx[i] = dy_i;
      if (alpha_requires_grad) { alpha_diff[i] = zero_val; }
    } else {
      dx[i] = dy_i * alpha[alpha_i];
      if (alpha_requires_grad) { alpha_diff[i] = dy_i * x_i; }
    }
  }
}

template<typename T, typename IndexType, int pack_size, bool alpha_requires_grad>
__global__ void PReluBackwardMultiAlphaGpu(const IndexType elem_cnt, const IndexType alpha_size,
                                           const IndexType inner_size, const T* x, const T* alpha,
                                           const T* dy, T* dx, T* alpha_diff) {
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  using LoadType = cuda::elementwise::PackType<T, pack_size>;
  using LoadPack = cuda::elementwise::Pack<T, pack_size>;
  T zero_val = static_cast<T>(0);
  for (int64_t linear_index = global_thread_id * pack_size; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x * pack_size) {
    IndexType alpha_idx = (linear_index / inner_size) % alpha_size;

    const LoadType* x_load = reinterpret_cast<const LoadType*>(x + linear_index);
    LoadPack x_vec;
    x_vec.storage = *x_load;

    const LoadType* dy_load = reinterpret_cast<const LoadType*>(dy + linear_index);
    LoadPack dy_vec;
    dy_vec.storage = *dy_load;

    LoadPack dx_vec;
    T alpha_val = alpha[alpha_idx];
    if (alpha_requires_grad) {
      LoadPack dalpha_vec;
      T zero_val = static_cast<T>(0.0);
#pragma unroll
      for (int i = 0; i < pack_size; i++) {
        if (x_vec.elem[i] > zero_val) {
          dx_vec.elem[i] = dy_vec.elem[i];
          dalpha_vec.elem[i] = zero_val;
        } else {
          dx_vec.elem[i] = dy_vec.elem[i] * alpha_val;
          dalpha_vec.elem[i] = dy_vec.elem[i] * x_vec.elem[i];
        }
      }
      *(reinterpret_cast<LoadType*>(dx + linear_index)) = dx_vec.storage;
      *(reinterpret_cast<LoadType*>(alpha_diff + linear_index)) = dalpha_vec.storage;
    } else {
#pragma unroll
      for (int i = 0; i < pack_size; i++) {
        if (x_vec.elem[i] > zero_val) {
          dx_vec.elem[i] = dy_vec.elem[i];
        } else {
          dx_vec.elem[i] = dy_vec.elem[i] * alpha_val;
        }
      }
      *(reinterpret_cast<LoadType*>(dx + linear_index)) = dx_vec.storage;
    }
  }
}

constexpr int32_t kBlockSize = 256;

template<typename T>
int GetLaunchPackSize(const int64_t inner_size) {
  constexpr int type_pack_size = cuda::elementwise::PackSize<T>();
  for (int launch_pack_size = 8; launch_pack_size > 0; launch_pack_size /= 2) {
    if (type_pack_size >= launch_pack_size && inner_size % launch_pack_size == 0) {
      return launch_pack_size;
    }
  }
  return 1;
}

template<typename T, typename IndexType>
void DispatchPreluForwardPackSize(ep::Stream* stream, const int64_t elem_cnt,
                                  const int64_t alpha_size, const int64_t inner_size, const T* x,
                                  const T* alpha, T* y) {
  int grid_size;
  const int pack_size = GetLaunchPackSize<T>(inner_size);
  const int64_t pack_num = elem_cnt / pack_size;
  cudaError_t err = cuda::elementwise::GetNumBlocks(pack_num, &grid_size);
  if (pack_size == 8) {
    PReluForwardMultiAlphaGpu<T, IndexType, 8>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            elem_cnt, alpha_size, inner_size, x, alpha, y);
  } else if (pack_size == 4) {
    PReluForwardMultiAlphaGpu<T, IndexType, 4>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            elem_cnt, alpha_size, inner_size, x, alpha, y);
  } else if (pack_size == 2) {
    PReluForwardMultiAlphaGpu<T, IndexType, 2>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            elem_cnt, alpha_size, inner_size, x, alpha, y);
  } else {
    BroadcastPReluMultiAlphaNaiveForwardGpu<T>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            elem_cnt, alpha_size, inner_size, x, alpha, y);
  }
}

template<typename T>
void DispatchPreluForwardIndex(ep::Stream* stream, const int64_t elem_cnt, const int64_t alpha_size,
                               const int64_t inner_size, const T* x, const T* alpha, T* y) {
  if (elem_cnt < GetMaxVal<int32_t>()) {
    DispatchPreluForwardPackSize<T, int32_t>(stream, elem_cnt, alpha_size, inner_size, x, alpha, y);
  } else {
    DispatchPreluForwardPackSize<T, int64_t>(stream, elem_cnt, alpha_size, inner_size, x, alpha, y);
  }
}

template<typename T, typename IndexType>
void DispatchPreluBackwardPackSize(ep::Stream* stream, const int64_t elem_cnt,
                                   const int64_t alpha_size, const int64_t inner_size, const T* x,
                                   const T* alpha, const T* dy, T* dx, T* alpha_diff,
                                   const bool alpha_requires_grad) {
  int grid_size;
  const int pack_size = GetLaunchPackSize<T>(inner_size);
  const int64_t pack_num = elem_cnt / pack_size;
  cudaError_t err = cuda::elementwise::GetNumBlocks(pack_num, &grid_size);

  if (pack_size == 8) {
    if (alpha_requires_grad) {
      PReluBackwardMultiAlphaGpu<T, IndexType, 8, true>
          <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              elem_cnt, alpha_size, inner_size, x, alpha, dy, dx, alpha_diff);
    } else {
      PReluBackwardMultiAlphaGpu<T, IndexType, 8, false>
          <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              elem_cnt, alpha_size, inner_size, x, alpha, dy, dx, alpha_diff);
    }
  } else if (pack_size == 4) {
    if (alpha_requires_grad) {
      PReluBackwardMultiAlphaGpu<T, IndexType, 4, true>
          <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              elem_cnt, alpha_size, inner_size, x, alpha, dy, dx, alpha_diff);
    } else {
      PReluBackwardMultiAlphaGpu<T, IndexType, 4, false>
          <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              elem_cnt, alpha_size, inner_size, x, alpha, dy, dx, alpha_diff);
    }
  } else if (pack_size == 2) {
    if (alpha_requires_grad) {
      PReluBackwardMultiAlphaGpu<T, IndexType, 2, true>
          <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              elem_cnt, alpha_size, inner_size, x, alpha, dy, dx, alpha_diff);
    } else {
      PReluBackwardMultiAlphaGpu<T, IndexType, 2, false>
          <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              elem_cnt, alpha_size, inner_size, x, alpha, dy, dx, alpha_diff);
    }

  } else {
    if (alpha_requires_grad) {
      BroadcastPReluMultiAlphaNaiveBackwardGpu<T, true>
          <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              elem_cnt, alpha_size, inner_size, x, alpha, dy, dx, alpha_diff);
    } else {
      BroadcastPReluMultiAlphaNaiveBackwardGpu<T, false>
          <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              elem_cnt, alpha_size, inner_size, x, alpha, dy, dx, alpha_diff);
    }
  }
}

template<typename T>
void DispatchPreluBackwardIndex(ep::Stream* stream, const int64_t elem_cnt,
                                const int64_t alpha_size, const int64_t inner_size, const T* x,
                                const T* alpha, const T* dy, T* dx, T* alpha_diff,
                                const bool alpha_requires_grad) {
  if (elem_cnt < GetMaxVal<int32_t>()) {
    DispatchPreluBackwardPackSize<T, int32_t>(stream, elem_cnt, alpha_size, inner_size, x, alpha,
                                              dy, dx, alpha_diff, alpha_requires_grad);
  } else {
    DispatchPreluBackwardPackSize<T, int64_t>(stream, elem_cnt, alpha_size, inner_size, x, alpha,
                                              dy, dx, alpha_diff, alpha_requires_grad);
  }
}

template<typename T, typename IndexType>
void DispatchPreluBackwardSingleAlphaTail(ep::Stream* stream, const IndexType elem_cnt, const T* x,
                                          const T* alpha, const T* dy, T* dx, T* alpha_diff,
                                          const bool alpha_requires_grad) {
  constexpr int pack_size = cuda::elementwise::PackSize<T>();
  const int64_t pack_num = elem_cnt / pack_size;
  int grid_size;
  cudaError_t err = cuda::elementwise::GetNumBlocks(pack_num, &grid_size);
  const int64_t tail_offset = pack_num * pack_size;
  const int64_t n_tail = elem_cnt - tail_offset;
  const bool tail = n_tail > 0 ? true : false;
  if (tail) {
    if (alpha_requires_grad) {
      PReluBackwardSingleAlphaGpu<T, IndexType, pack_size, true, true>
          <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              elem_cnt, n_tail, x, alpha, dy, dx, alpha_diff, x + tail_offset, dy + tail_offset,
              dx + tail_offset, alpha_diff + tail_offset);
    } else {
      PReluBackwardSingleAlphaGpu<T, IndexType, pack_size, true, false>
          <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              elem_cnt, n_tail, x, alpha, dy, dx, alpha_diff, x + tail_offset, dy + tail_offset,
              dx + tail_offset, alpha_diff + tail_offset);
    }
  } else {
    if (alpha_requires_grad) {
      PReluBackwardSingleAlphaGpu<T, IndexType, pack_size, false, true>
          <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              elem_cnt, n_tail, x, alpha, dy, dx, alpha_diff, x + tail_offset, dy + tail_offset,
              dx + tail_offset, alpha_diff + tail_offset);
    } else {
      PReluBackwardSingleAlphaGpu<T, IndexType, pack_size, false, false>
          <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
              elem_cnt, n_tail, x, alpha, dy, dx, alpha_diff, x + tail_offset, dy + tail_offset,
              dx + tail_offset, alpha_diff + tail_offset);
    }
  }
}

template<typename T>
void DispatchPreluBackwardSingleAlphaIndex(ep::Stream* stream, const int64_t elem_cnt, const T* x,
                                           const T* alpha, const T* dy, T* dx, T* alpha_diff,
                                           const bool alpha_requires_grad) {
  if (elem_cnt < GetMaxVal<int32_t>()) {
    DispatchPreluBackwardSingleAlphaTail<T, int32_t>(stream, elem_cnt, x, alpha, dy, dx, alpha_diff,
                                                     alpha_requires_grad);
  } else {
    DispatchPreluBackwardSingleAlphaTail<T, int64_t>(stream, elem_cnt, x, alpha, dy, dx, alpha_diff,
                                                     alpha_requires_grad);
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
    const int32_t elem_cnt = x->shape_view().elem_cnt();
    const int32_t batch = x->shape_view().At(0);
    const int32_t channels = (x->shape_view().NumAxes() == 1) ? 1 : x->shape_view().At(1);
    const int32_t alpha_size = alpha->shape_view().elem_cnt();
    const int32_t inner_size = elem_cnt / batch / channels;

    if (alpha_size == 1) {
      OF_CUDA_CHECK((cuda::elementwise::UnaryWithFactory(
          PreluForwardSingleAlphaPtrFunctor<T>(reinterpret_cast<const T*>(alpha->dptr())), elem_cnt,
          reinterpret_cast<T*>(y->mut_dptr()), reinterpret_cast<const T*>(x->dptr()),
          ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
    } else {
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
    const bool alpha_requires_grad = ctx->Attr<bool>("alpha_requires_grad");
    const int32_t elem_cnt = x->shape_view().elem_cnt();
    T* broadcasted_alpha_diff = tmp_buffer->mut_dptr<T>();
    T* reduce_sum_tmp_buf = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>()
                                                 + GetCudaAlignedSize(elem_cnt * sizeof(T)));

    const int32_t batch = x->shape_view().At(0);
    const int32_t channels = (x->shape_view().NumAxes() == 1) ? 1 : x->shape_view().At(1);
    const int32_t alpha_size = alpha->shape_view().elem_cnt();
    const int32_t inner_size = elem_cnt / batch / channels;

    const Shape& left_extended_shape =
        CreatePreluLeftExtendedShape(ShapeView(x->shape_view()), alpha_size);
    if (alpha_size == 1) {
      DispatchPreluBackwardSingleAlphaIndex<T>(ctx->stream(), elem_cnt, x->dptr<T>(),
                                               alpha->dptr<T>(), dy->dptr<T>(), dx->mut_dptr<T>(),
                                               broadcasted_alpha_diff, alpha_requires_grad);
    } else {
      DispatchPreluBackwardIndex<T>(ctx->stream(), elem_cnt, alpha_size, inner_size, x->dptr<T>(),
                                    alpha->dptr<T>(), dy->dptr<T>(), dx->mut_dptr<T>(),
                                    broadcasted_alpha_diff, alpha_requires_grad);
    }
    if (alpha_requires_grad) {
      NdarrayUtil<DeviceType::kCUDA, T>::ReduceSum(
          ctx->stream(), XpuVarNdarray<T>(left_extended_shape, alpha_diff->mut_dptr<T>()),
          XpuVarNdarray<const T>(x->shape_view(), broadcasted_alpha_diff),
          XpuVarNdarray<T>(x->shape_view(), reduce_sum_tmp_buf));
    }
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
