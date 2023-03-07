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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

template<typename T, int N>
struct FusedRotaryEmbeddingParam {
    const T* x;
    const T* cos;
    const T* sin;
    T* out;
    const int32_t num_rows;
    int32_t num_elements;
    size_t x_stride[N];
    size_t sinuous_stride[N];
    size_t sinuous_mask[N];


    explicit FusedRotaryEmbeddingParam(const T* x, const T* cos, const T* sin, T* out, 
      const int32_t num_rows, const int32_t num_elements): 
        x(x), cos(cos), sin(sin), out(out), num_rows(num_rows), num_elements(num_elements) {}
};

template<typename T, typename IndexType, size_t pack_size, size_t num_dims>
__global__ void FusedRotaryEmbeddingComputeKernel(FusedRotaryEmbeddingParam<T, num_dims> param) {
  const T* x = param.x;
  const T* cos = param.cos;
  const T* sin = param.sin;
  T* out = param.out;
  const IndexType num_elements = param.num_elements;
  IndexType index[num_dims];
  for (IndexType offset = threadIdx.x + blockIdx.x * blockDim.x; offset < num_elements; offset += blockDim.x * gridDim.x) {
    using LoadPack = cuda::elementwise::Packed<T, pack_size>;
    IndexType packed_offset = offset * pack_size;
    const LoadPack* x_load = reinterpret_cast<const LoadPack*>(x + packed_offset);
    const LoadPack x_vec = *x_load;

#pragma unloop
    for (int i = 0; i < num_dims; i++) {
      index[i] = packed_offset / param.x_stride[i];
      packed_offset = packed_offset - index[i] * param.x_stride[i];
    }

    IndexType sinuous_offset = 0;
#pragma unloop
    for (int i = 0; i < num_dims; i++) {
      sinuous_offset += (index[i] * param.sinuous_mask[i] * param.sinuous_stride[i]);
    }
    const LoadPack* cos_load = reinterpret_cast<const LoadPack*>(cos + sinuous_offset);
    const LoadPack* sin_load = reinterpret_cast<const LoadPack*>(sin + sinuous_offset);
    const LoadPack cos_vec = *cos_load, sin_vec = *sin_load;
    LoadPack out_vec;

#pragma unloop
    for (int i = 0; i < pack_size / 2; i++) {
      out_vec.elem[i*2] = x_vec.elem[i*2] * cos_vec.elem[i*2] - x_vec.elem[i*2 + 1] * sin_vec.elem[i*2];
      out_vec.elem[i*2 + 1] = x_vec.elem[i*2 + 1] * cos_vec.elem[i*2 + 1] + x_vec.elem[i*2] * sin_vec.elem[i*2 + 1];
    }

    *(reinterpret_cast<LoadPack*>(out + offset * pack_size)) = out_vec;
  }
}

template<typename T, typename IndexType, size_t pack_size, size_t num_dims>
void LaunchKernel(const T* x, const T* cos, const T* sin, T* out, const int64_t* x_shape, const int64_t* sinuous_shape,
  const int32_t num_rows, const int32_t num_elements, const std::string& layout) {
    DimVector kernel_x_shape(num_dims), kernel_sinuous_shape(num_dims);
    size_t x_stride[num_dims];
    size_t sinuous_stride[num_dims];


    x_stride[num_dims-1] = 1;
    sinuous_stride[num_dims-1] = 1;

    if (layout == "BHMK") {
#pragma unloop
      for (int i = 0; i < num_dims; i++) {
        kernel_x_shape.at(i) = x_shape[i];
      }
      kernel_sinuous_shape.at(0) = 1;
      kernel_sinuous_shape.at(1) = 1;
      kernel_sinuous_shape.at(2) = sinuous_shape[0];
      kernel_sinuous_shape.at(3) = sinuous_shape[1];
    } else if (layout == "BME") {
      kernel_x_shape.at(0) = x_shape[0];
      kernel_x_shape.at(1) = x_shape[1];
      kernel_x_shape.at(2) = x_shape[2] / sinuous_shape[1];
      kernel_x_shape.at(3) = sinuous_shape[1];
      kernel_sinuous_shape.at(0) = 1;
      kernel_sinuous_shape.at(1) = sinuous_shape[0];
      kernel_sinuous_shape.at(2) = 1;
      kernel_sinuous_shape.at(3) = sinuous_shape[1];
    }

    for (int i = num_dims-2; i >= 0; i--) {
      x_stride[i] = x_stride[i+1] * kernel_x_shape.at(i+1);
      sinuous_stride[i] = sinuous_stride[i+1] * kernel_sinuous_shape.at(i+1);
    }

    struct FusedRotaryEmbeddingParam<T, num_dims> param(reinterpret_cast<const T*>(x), reinterpret_cast<const T*>(cos), 
      reinterpret_cast<const T*>(sin), reinterpret_cast<T*>(out), num_rows, num_elements);

#pragma unloop
    for (int i = 0; i < num_dims; i++) {
      param.x_stride[i] = x_stride[i];
      param.sinuous_mask[i] = (kernel_sinuous_shape.at(i) == 1) ? 0 : 1;
      param.sinuous_stride[i] = sinuous_stride[i];
    }

    constexpr size_t blk_size = 128;
    FusedRotaryEmbeddingComputeKernel<T, IndexType, pack_size, num_dims><<<(param.num_elements + blk_size - 1)/blk_size, blk_size>>>
      (param);
}

template<typename T, typename IndexType, size_t num_dims>
void DispatchPackSize(FusedRotaryEmbeddingParam<T, num_dims>& param, const int64_t* x_shape, const int64_t* sinuous_shape, 
  const std::string& layout) {
  const size_t blk_size = 128;
  const T* x = param.x;
  const IndexType num_rows = param.num_rows;

  const auto CheckPackSize = [&](const size_t pack_size) {
    bool r = (((reinterpret_cast<uintptr_t>(x) % (sizeof(T) * pack_size)) == 0) && ((num_rows % pack_size) == 0) && ((16 / sizeof(T)) >= pack_size));
    return r;
  };

  if (CheckPackSize(8)) {
    param.num_elements /= 8;
    LaunchKernel<T, IndexType, 8, num_dims>(param.x, param.cos, param.sin, param.out, x_shape, sinuous_shape, param.num_rows, param.num_elements, layout);
  } else if (CheckPackSize(4)) {
    param.num_elements /= 4;
    LaunchKernel<T, IndexType, 4, num_dims>(param.x, param.cos, param.sin, param.out, x_shape, sinuous_shape, param.num_rows, param.num_elements, layout);
  } else {
    param.num_elements /= 2;
    LaunchKernel<T, IndexType, 2, num_dims>(param.x, param.cos, param.sin, param.out, x_shape, sinuous_shape, param.num_rows, param.num_elements, layout);
  }
  
}

template<typename T, size_t num_dims>
void DispatchIndex(FusedRotaryEmbeddingParam<T, num_dims>& param, const int64_t* x_shape, const int64_t* sinuous_shape, const std::string& layout) {
  if (param.num_elements < (1 << 30)) {
    DispatchPackSize<T, int32_t, num_dims>(param, x_shape, sinuous_shape, layout);
  } else {
    DispatchPackSize<T, int64_t, num_dims>(param, x_shape, sinuous_shape, layout);
  }
}

template<typename T>
class FusedRotaryEmbeddingKernel final : public user_op::OpKernel {
 public:
  FusedRotaryEmbeddingKernel() = default;
  ~FusedRotaryEmbeddingKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();

    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* cos = ctx->Tensor4ArgNameAndIndex("cos", 0);
    const user_op::Tensor* sin = ctx->Tensor4ArgNameAndIndex("sin", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const std::string& layout = ctx->Attr<std::string>("layout");

    const DataType data_type = out->data_type();
    void* y_ptr = out->mut_dptr();

    constexpr size_t N = 4;

    DimVector x_shape(N), sinuous_shape(N);
    size_t x_stride[N];
    size_t sinuous_stride[N];


    x_stride[N-1] = 1;
    sinuous_stride[N-1] = 1;

    if (layout == "BHMK") {
#pragma unloop
      for (int i = 0; i < N; i++) {
        x_shape.at(i) = x->shape_view().At(i);
      }
      sinuous_shape.at(0) = 1;
      sinuous_shape.at(1) = 1;
      sinuous_shape.at(2) = cos->shape_view().At(0);
      sinuous_shape.at(3) = cos->shape_view().At(1);
    } else if (layout == "BME") {
      x_shape.at(0) = x->shape_view().At(0);
      x_shape.at(1) = x->shape_view().At(1);
      x_shape.at(2) = x->shape_view().At(2) / cos->shape_view().At(1);
      x_shape.at(3) = cos->shape_view().At(1);
      sinuous_shape.at(0) = 1;
      sinuous_shape.at(1) = cos->shape_view().At(0);
      sinuous_shape.at(2) = 1;
      sinuous_shape.at(3) = cos->shape_view().At(1);
    }

    for (int i = N-2; i >= 0; i--) {
      x_stride[i] = x_stride[i+1] * x_shape.at(i+1);
      sinuous_stride[i] = sinuous_stride[i+1] * sinuous_shape.at(i+1);
    }    

    //TODO: index type should change up to the size of num_elements
    const int32_t num_rows = x_shape.at(3);
    int32_t num_elements = x->shape_view().Count(0, x->shape_view().NumAxes()); //TODO: because helper uses int32_t, so no size_t

    //TODO: hard code num_dims & seems redundant template problem...
    struct FusedRotaryEmbeddingParam<T, N> param(reinterpret_cast<const T*>(x->dptr()), reinterpret_cast<const T*>(cos->dptr()), 
            reinterpret_cast<const T*>(sin->dptr()), reinterpret_cast<T*>(out->mut_dptr()), num_rows, num_elements);

#pragma unloop
    for (int i = 0; i < N; i++) {
      param.x_stride[i] = x_stride[i];
      param.sinuous_mask[i] = (sinuous_shape.at(i) == 1) ? 0 : 1;
      param.sinuous_stride[i] = sinuous_stride[i];
    }
    DispatchIndex<T, N>(param, x->shape_view().data(), cos->shape_view().data(), layout);
/*
    FusedRotaryEmbeddingComputeKernel<T, 4, 4><<<(num_elements + blk_size - 1)/blk_size, blk_size>>>
            (reinterpret_cast<const T*>(x->dptr()), reinterpret_cast<const T*>(cos->dptr()), 
            reinterpret_cast<const T*>(sin->dptr()), reinterpret_cast<T*>(out->mut_dptr()),
            num_elements, param);*/
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_ROTARY_EMBEDDING_KERNEL_GPU(dtype)               \
  REGISTER_USER_KERNEL("fused_rotary_embedding")                            \
      .SetCreateFn<FusedRotaryEmbeddingKernel<dtype>>()                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_ROTARY_EMBEDDING_KERNEL_GPU(float);
REGISTER_FUSED_ROTARY_EMBEDDING_KERNEL_GPU(half);
#if CUDA_VERSION >= 11000
REGISTER_FUSED_ROTARY_EMBEDDING_KERNEL_GPU(nv_bfloat16);
#endif  // CUDA_VERSION >= 11000

}  // namespace

}  // namespace oneflow
