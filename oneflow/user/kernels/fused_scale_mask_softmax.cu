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
#include "oneflow/core/cuda/softmax.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

template<typename SRC, typename DST, size_t num_dims>
struct ScaleMaskLoad {
  ScaleMaskLoad(const SRC* src, const int8_t* mask, const int64_t row_size, 
                const float fill, const float scale, const int64_t* mask_dims,
                NdIndexOffsetHelper<int64_t, num_dims> input_index_helper, 
                NdIndexOffsetHelper<int64_t, num_dims> mask_index_helper)
      : src(src), mask(mask), row_size(row_size), fill(fill), scale(scale), mask_dims(mask_dims),
        input_index_helper(input_index_helper), mask_index_helper(mask_index_helper) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) {
    cuda::softmax::Pack<SRC, N> pack;
    cuda::softmax::Pack<int8_t, N> mask_pack;
    const int64_t offset = row * row_size + col;
    int64_t input_index[num_dims];
    int64_t mask_index[num_dims];
    input_index_helper.OffsetToNdIndex(offset, input_index);
    for (int dim = 0; dim < num_dims; ++dim) {
      if (mask_dims[dim] == 1) {
        mask_index[dim] = 0;
      } else {
        mask_index[dim] = input_index[dim];
      }
    }
    const int64_t mask_offset = mask_index_helper.NdIndexToOffset(mask_index);
    pack.storage = *reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(src + offset);
    mask_pack.storage = *reinterpret_cast<const cuda::softmax::PackType<int8_t, N>*>(mask + mask_offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (mask_pack.elem[i] == 0) {
        dst[i] = static_cast<DST>(fill);
      } else {
        dst[i] = static_cast<DST>(pack.elem[i]) * static_cast<DST>(scale);
      }
    }
  }
  const SRC* src;
  const int8_t* mask;
  const int64_t row_size;
  const float fill;
  const float scale;
  const int64_t* mask_dims;
  NdIndexOffsetHelper<int64_t, num_dims> input_index_helper;
  NdIndexOffsetHelper<int64_t, num_dims> mask_index_helper;
};

template<typename SRC, typename DST, size_t num_dims>
struct ScaleMaskStore {
  ScaleMaskStore(DST* dst, const int8_t* mask, const int64_t row_size, 
                const float fill, const float scale, const int64_t* mask_dims,
                NdIndexOffsetHelper<int64_t, num_dims> input_index_helper, 
                NdIndexOffsetHelper<int64_t, num_dims> mask_index_helper)
      : dst(dst), mask(mask), row_size(row_size), fill(fill), scale(scale), mask_dims(mask_dims),
        input_index_helper(input_index_helper), mask_index_helper(mask_index_helper) {}
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    cuda::softmax::Pack<DST, N> pack;
    cuda::softmax::Pack<int8_t, N> mask_pack;
    const int64_t offset = row * row_size + col;
    int64_t input_index[num_dims];
    int64_t mask_index[num_dims];
    input_index_helper.OffsetToNdIndex(offset, input_index);
    for (int dim = 0; dim < num_dims; ++dim) {
      if (mask_dims[dim] == 1) {
        mask_index[dim] = 0;
      } else {
        mask_index[dim] = input_index[dim];
      }
    }
    const int64_t mask_offset = mask_index_helper.NdIndexToOffset(mask_index);
    mask_pack.storage = *reinterpret_cast<const cuda::softmax::PackType<int8_t, N>*>(mask + mask_offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (mask_pack.elem[i] == 0) {
        pack.elem[i] = static_cast<DST>(fill);
      } else {
        pack.elem[i] = static_cast<DST>(src[i]) * static_cast<DST>(scale);
      }
    }
    *reinterpret_cast<cuda::softmax::PackType<DST, N>*>(dst + offset) = pack.storage;
  }
  DST* dst;
  const int8_t* mask;
  const int64_t row_size;
  const float fill;
  const float scale;
  const int64_t* mask_dims;
  NdIndexOffsetHelper<int64_t, num_dims> input_index_helper;
  NdIndexOffsetHelper<int64_t, num_dims> mask_index_helper;
};

template<typename SRC, typename DST, size_t num_dims>
ScaleMaskLoad<SRC, DST, num_dims> MakeScaleMaskLoad(const SRC* src, const int8_t* mask, 
                                                    const int64_t row_size, 
                                                    const float fill, const float scale,
                                                    const int64_t* input_dims,
                                                    const int64_t* mask_dims) {
  NdIndexOffsetHelper<int64_t, num_dims> input_index_helper(input_dims);
  NdIndexOffsetHelper<int64_t, num_dims> mask_index_helper(mask_dims);
  ScaleMaskLoad<SRC, DST, num_dims> load(src, mask, row_size, fill, scale, mask_dims, 
                                         input_index_helper, mask_index_helper);
  return load;
}

template<typename SRC, typename DST, size_t num_dims>
ScaleMaskStore<SRC, DST, num_dims> MakeScaleMaskStore(DST* dst, const int8_t* mask, 
                                                      const int64_t row_size, 
                                                      const float fill, const float scale,
                                                      const int64_t* input_dims,
                                                      const int64_t* mask_dims) {
  NdIndexOffsetHelper<int64_t, num_dims> input_index_helper(input_dims);
  NdIndexOffsetHelper<int64_t, num_dims> mask_index_helper(mask_dims);
  ScaleMaskStore<SRC, DST, num_dims> store(dst, mask, row_size, fill, scale, mask_dims, 
                                           input_index_helper, mask_index_helper);
  return store;
}

template<typename T>
class FusedScaleMaskSoftmaxKernel final : public user_op::OpKernel {
 public:
  FusedScaleMaskSoftmaxKernel() = default;
  ~FusedScaleMaskSoftmaxKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const ShapeView& x_shape = x->shape();
    const ShapeView& mask_shape = mask->shape();
    CHECK_GE(x_shape.NumAxes(), 2);
    const int64_t cols = x_shape.At(x_shape.NumAxes() - 1);
    const int64_t rows = x_shape.Count(0, x_shape.NumAxes() - 1);
    const size_t num_dims = x_shape.NumAxes() - 1;
    const int64_t* input_dims = x_shape.ptr();
    const int64_t* mask_dims = mask_shape.ptr();
    using ComputeType = typename cuda::softmax::DefaultComputeType<T>::type;
    if (num_dims == 2) {
      NdIndexOffsetHelper<int64_t, 2> input_index_helper(input_dims);
      NdIndexOffsetHelper<int64_t, 2> mask_index_helper(mask_dims);
      ScaleMaskLoad<T, ComputeType, 2> load(x->dptr<T>(), mask->dptr<int8_t>(), cols,
                                            ctx->Attr<float>("mask_fill_value"),
                                            ctx->Attr<float>("scale_value"),
                                            mask_dims, input_index_helper, mask_index_helper);
      cuda::softmax::DirectStore<ComputeType, T> store(y->mut_dptr<T>(), cols);
      OF_CUDA_CHECK((cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
          ctx->stream()->As<ep::CudaStream>()->cuda_stream(), load, store, rows, cols)));
    } else if (num_dims == 3) {
      NdIndexOffsetHelper<int64_t, 3> input_index_helper(input_dims);
      NdIndexOffsetHelper<int64_t, 3> mask_index_helper(mask_dims);
      ScaleMaskLoad<T, ComputeType, 3> load(x->dptr<T>(), mask->dptr<int8_t>(), cols,
                                      ctx->Attr<float>("mask_fill_value"),
                                      ctx->Attr<float>("scale_value"),
                                      mask_dims, input_index_helper, mask_index_helper);
      cuda::softmax::DirectStore<ComputeType, T> store(y->mut_dptr<T>(), cols);
      OF_CUDA_CHECK((cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
          ctx->stream()->As<ep::CudaStream>()->cuda_stream(), load, store, rows, cols)));
    } else if (num_dims == 4) {
      NdIndexOffsetHelper<int64_t, 4> input_index_helper(input_dims);
      NdIndexOffsetHelper<int64_t, 4> mask_index_helper(mask_dims);
      ScaleMaskLoad<T, ComputeType, 4> load(x->dptr<T>(), mask->dptr<int8_t>(), cols,
                                            ctx->Attr<float>("mask_fill_value"),
                                            ctx->Attr<float>("scale_value"),
                                            mask_dims, input_index_helper, mask_index_helper);
      cuda::softmax::DirectStore<ComputeType, T> store(y->mut_dptr<T>(), cols);
      OF_CUDA_CHECK((cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
          ctx->stream()->As<ep::CudaStream>()->cuda_stream(), load, store, rows, cols)));
    } else if (num_dims == 5) {
      NdIndexOffsetHelper<int64_t, 5> input_index_helper(input_dims);
      NdIndexOffsetHelper<int64_t, 5> mask_index_helper(mask_dims);
      ScaleMaskLoad<T, ComputeType, 5> load(x->dptr<T>(), mask->dptr<int8_t>(), cols,
                                            ctx->Attr<float>("mask_fill_value"),
                                            ctx->Attr<float>("scale_value"),
                                            mask_dims, input_index_helper, mask_index_helper);
      cuda::softmax::DirectStore<ComputeType, T> store(y->mut_dptr<T>(), cols);
      OF_CUDA_CHECK((cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
          ctx->stream()->As<ep::CudaStream>()->cuda_stream(), load, store, rows, cols)));
    } else {
      UNIMPLEMENTED();
      NdIndexOffsetHelper<int64_t, 1> input_index_helper(input_dims);
      NdIndexOffsetHelper<int64_t, 1> mask_index_helper(mask_dims);
      ScaleMaskLoad<T, ComputeType, 1> load(x->dptr<T>(), mask->dptr<int8_t>(), cols,
                                            ctx->Attr<float>("mask_fill_value"),
                                            ctx->Attr<float>("scale_value"),
                                            mask_dims, input_index_helper, mask_index_helper);
      cuda::softmax::DirectStore<ComputeType, T> store(y->mut_dptr<T>(), cols);
      OF_CUDA_CHECK((cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
          ctx->stream()->As<ep::CudaStream>()->cuda_stream(), load, store, rows, cols)));
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUCED_SCALE_MASK_SOFTMAX_CUDA_KERNEL(dtype)           \
  REGISTER_USER_KERNEL("fused_scale_mask_softmax")                     \
      .SetCreateFn<FusedScaleMaskSoftmaxKernel<dtype>>()               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_FUCED_SCALE_MASK_SOFTMAX_CUDA_KERNEL(half)
REGISTER_FUCED_SCALE_MASK_SOFTMAX_CUDA_KERNEL(float)
REGISTER_FUCED_SCALE_MASK_SOFTMAX_CUDA_KERNEL(double)
#undef REGISTER_FUCED_SCALE_MASK_SOFTMAX_CUDA_KERNEL

template<typename T>
class FusedScaleMaskSoftmaxGradKernel final : public user_op::OpKernel {
 public:
  FusedScaleMaskSoftmaxGradKernel() = default;
  ~FusedScaleMaskSoftmaxGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const ShapeView& dy_shape = dy->shape();
    const ShapeView& mask_shape = mask->shape();
    CHECK_GE(dy_shape.NumAxes(), 2);
    const int64_t cols = dy_shape.At(dy_shape.NumAxes() - 1);
    const int64_t rows = dy_shape.Count(0, dy_shape.NumAxes() - 1);
    const size_t num_dims = dy_shape.NumAxes() - 1;
    const int64_t* input_dims = dy_shape.ptr();
    const int64_t* mask_dims = mask_shape.ptr();
    using ComputeType = typename cuda::softmax::DefaultComputeType<T>::type;
    cuda::softmax::DirectLoad<T, ComputeType> load_y(y->dptr<T>(), cols);
    cuda::softmax::DirectLoad<T, ComputeType> load_dy(dy->dptr<T>(), cols);
    if (num_dims == 2) {
      NdIndexOffsetHelper<int64_t, 2> input_index_helper(input_dims);
      NdIndexOffsetHelper<int64_t, 2> mask_index_helper(mask_dims);
      ScaleMaskStore<ComputeType, T, 2> store(dx->mut_dptr<T>(), mask->dptr<int8_t>(), cols,
                                              static_cast<float>(0.0), ctx->Attr<float>("scale_value"), 
                                              mask_dims, input_index_helper, mask_index_helper);
      OF_CUDA_CHECK((cuda::softmax::DispatchSoftmaxGrad<decltype(load_y), decltype(load_dy),
                                                      decltype(store), ComputeType>(
         ctx->stream()->As<ep::CudaStream>()->cuda_stream(), load_y, load_dy, store, rows, cols)));
    } else if (num_dims == 3) {
      NdIndexOffsetHelper<int64_t, 3> input_index_helper(input_dims);
      NdIndexOffsetHelper<int64_t, 3> mask_index_helper(mask_dims);
      ScaleMaskStore<ComputeType, T, 3> store(dx->mut_dptr<T>(), mask->dptr<int8_t>(), cols,
                                              static_cast<float>(0.0), ctx->Attr<float>("scale_value"), 
                                              mask_dims, input_index_helper, mask_index_helper);
      OF_CUDA_CHECK((cuda::softmax::DispatchSoftmaxGrad<decltype(load_y), decltype(load_dy),
                                                      decltype(store), ComputeType>(
         ctx->stream()->As<ep::CudaStream>()->cuda_stream(), load_y, load_dy, store, rows, cols)));
    } else if (num_dims == 4) {
      NdIndexOffsetHelper<int64_t, 4> input_index_helper(input_dims);
      NdIndexOffsetHelper<int64_t, 4> mask_index_helper(mask_dims);
      ScaleMaskStore<ComputeType, T, 4> store(dx->mut_dptr<T>(), mask->dptr<int8_t>(), cols,
                                              static_cast<float>(0.0), ctx->Attr<float>("scale_value"), 
                                              mask_dims, input_index_helper, mask_index_helper);
      OF_CUDA_CHECK((cuda::softmax::DispatchSoftmaxGrad<decltype(load_y), decltype(load_dy),
                                                      decltype(store), ComputeType>(
         ctx->stream()->As<ep::CudaStream>()->cuda_stream(), load_y, load_dy, store, rows, cols)));
    } else if (num_dims == 5) {
      NdIndexOffsetHelper<int64_t, 5> input_index_helper(input_dims);
      NdIndexOffsetHelper<int64_t, 5> mask_index_helper(mask_dims);
      ScaleMaskStore<ComputeType, T, 5> store(dx->mut_dptr<T>(), mask->dptr<int8_t>(), cols,
                                              static_cast<float>(0.0), ctx->Attr<float>("scale_value"), 
                                              mask_dims, input_index_helper, mask_index_helper);
      OF_CUDA_CHECK((cuda::softmax::DispatchSoftmaxGrad<decltype(load_y), decltype(load_dy),
                                                      decltype(store), ComputeType>(
         ctx->stream()->As<ep::CudaStream>()->cuda_stream(), load_y, load_dy, store, rows, cols)));
    } else {
      UNIMPLEMENTED();
      NdIndexOffsetHelper<int64_t, 1> input_index_helper(input_dims);
      NdIndexOffsetHelper<int64_t, 1> mask_index_helper(mask_dims);
      ScaleMaskStore<ComputeType, T, 1> store(dx->mut_dptr<T>(), mask->dptr<int8_t>(), cols,
                                              static_cast<float>(0.0), ctx->Attr<float>("scale_value"), 
                                              mask_dims, input_index_helper, mask_index_helper);
      OF_CUDA_CHECK((cuda::softmax::DispatchSoftmaxGrad<decltype(load_y), decltype(load_dy),
                                                      decltype(store), ComputeType>(
         ctx->stream()->As<ep::CudaStream>()->cuda_stream(), load_y, load_dy, store, rows, cols)));
    } 
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUCED_SCALE_MASK_SOFTMAX_GRAD_KERNEL(dtype)           \
  REGISTER_USER_KERNEL("fused_scale_mask_softmax_grad")                \
      .SetCreateFn<FusedScaleMaskSoftmaxGradKernel<dtype>>()           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_FUCED_SCALE_MASK_SOFTMAX_GRAD_KERNEL(half)
REGISTER_FUCED_SCALE_MASK_SOFTMAX_GRAD_KERNEL(float)
REGISTER_FUCED_SCALE_MASK_SOFTMAX_GRAD_KERNEL(double)
#undef REGISTER_FUCED_SCALE_MASK_SOFTMAX_GRAD_KERNEL

}  // namespace oneflow
