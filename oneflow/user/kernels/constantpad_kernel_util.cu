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
#include <cstdint>
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/constantpad_kernel_util.h"

namespace oneflow {
namespace user_op {

template<typename IN_T>
__global__ void DoCUDAConstantPad1d(const IN_T* src, IN_T* dest,
                                    const NdIndexOffsetHelper<int64_t, 3> index_helper,
                                    int64_t elem_num, int64_t n_channel, int64_t y_width,
                                    int64_t x_width, int64_t pad_left, const IN_T const_value) {
  DoConstantPad1d<IN_T>(src, dest, index_helper, elem_num, n_channel, y_width, x_width, pad_left,
                        const_value);
};

template<typename IN_T>
__global__ void DoCUDAConstantPad3d(const IN_T* src, IN_T* dest,
                                    const NdIndexOffsetHelper<int64_t, 5> index_helper,
                                    int64_t elem_num, int64_t n_channel, int64_t y_depth,
                                    int64_t y_height, int64_t y_width, int64_t x_depth,
                                    int64_t x_height, int64_t x_width, int64_t pad_front,
                                    int64_t pad_left, int64_t pad_top, const IN_T const_value) {
  DoConstantPad3d<IN_T>(src, dest, index_helper, elem_num, n_channel, y_depth, y_height, y_width,
                        x_depth, x_height, x_width, pad_front, pad_left, pad_top, const_value);
};

template<typename IN_T>
__global__ void DoCUDAConstantPad1dGrad(const IN_T* src, IN_T* dest,
                                        const NdIndexOffsetHelper<int64_t, 3> index_helper,
                                        int64_t elem_num, int64_t n_channel, int64_t dy_width,
                                        int64_t dx_width, int64_t pad_left) {
  DoConstantPad1dGrad<IN_T>(src, dest, index_helper, elem_num, n_channel, dy_width, dx_width,
                            pad_left);
};

template<typename IN_T>
__global__ void DoCUDAConstantPad3dGrad(const IN_T* src, IN_T* dest,
                                        const NdIndexOffsetHelper<int64_t, 5> index_helper,
                                        int64_t elem_num, int64_t n_channel, int64_t dy_depth,
                                        int64_t dy_height, int64_t dy_width, int64_t dx_depth,
                                        int64_t dx_height, int64_t dx_width, int64_t pad_front,
                                        int64_t pad_left, int64_t pad_top) {
  DoConstantPad3dGrad<IN_T>(src, dest, index_helper, elem_num, n_channel, dy_depth, dy_height,
                            dy_width, dx_height, dx_depth, dx_width, pad_front, pad_left, pad_top);
};

template<typename IN_T>
struct ConstantPad1dFunctor<DeviceType::kGPU, IN_T> final {
  void operator()(DeviceCtx* ctx, const IN_T* src, IN_T* dest,
                  const NdIndexOffsetHelper<int64_t, 3>& index_helper, const ShapeView& x_shape,
                  const ShapeView& y_shape, const std::vector<int64_t>& padding,
                  IN_T constant_value) {
    const int64_t c_idx = 1;
    const int64_t w_idx = 2;

    DoCUDAConstantPad1d<IN_T>
        <<<BlocksNum4ThreadsNum(y_shape.Count(0)), kCudaThreadsNumPerBlock, 0,
           ctx->cuda_stream()>>>(src, dest, index_helper, y_shape.Count(0), y_shape.At(c_idx),
                                 y_shape.At(w_idx), x_shape.At(w_idx), padding[0], constant_value);
  }
};

// float16 implementation
template<>
void ConstantPad1dFunctor<DeviceType::kGPU, float16>::operator()(
    DeviceCtx* ctx, const float16* src, float16* dest,
    const NdIndexOffsetHelper<int64_t, 3>& index_helper, const ShapeView& x_shape,
    const ShapeView& y_shape, const std::vector<int64_t>& padding, float16 constant_value) {
  const int64_t c_idx = 1;
  const int64_t w_idx = 2;
  DoCUDAConstantPad1d<half>
      <<<BlocksNum4ThreadsNum(y_shape.Count(0)), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          reinterpret_cast<const half*>(src), reinterpret_cast<half*>(dest), index_helper,
          y_shape.Count(0), y_shape.At(c_idx), y_shape.At(w_idx), x_shape.At(w_idx), padding[0],
          static_cast<const half>(constant_value));
}

template<typename IN_T>
struct ConstantPad3dFunctor<DeviceType::kGPU, IN_T> final {
  void operator()(DeviceCtx* ctx, const IN_T* src, IN_T* dest,
                  const NdIndexOffsetHelper<int64_t, 5>& index_helper, const ShapeView& x_shape,
                  const ShapeView& y_shape, const std::vector<int64_t>& padding,
                  IN_T constant_value) {
    const int64_t c_idx = 1;
    const int64_t d_idx = 2;
    const int64_t h_idx = 3;
    const int64_t w_idx = 4;

    DoCUDAConstantPad3d<IN_T><<<BlocksNum4ThreadsNum(y_shape.Count(0)), kCudaThreadsNumPerBlock, 0,
                                ctx->cuda_stream()>>>(
        src, dest, index_helper, y_shape.Count(0), y_shape.At(c_idx), y_shape.At(d_idx),
        y_shape.At(h_idx), y_shape.At(w_idx), x_shape.At(d_idx), x_shape.At(h_idx),
        x_shape.At(w_idx), padding[4], padding[0], padding[2], constant_value);
  }
};

// float16 implementation
template<>
void ConstantPad3dFunctor<DeviceType::kGPU, float16>::operator()(
    DeviceCtx* ctx, const float16* src, float16* dest,
    const NdIndexOffsetHelper<int64_t, 5>& index_helper, const ShapeView& x_shape,
    const ShapeView& y_shape, const std::vector<int64_t>& padding, float16 constant_value) {
  const int64_t c_idx = 1;
  const int64_t d_idx = 2;
  const int64_t h_idx = 3;
  const int64_t w_idx = 4;
  DoCUDAConstantPad3d<half>
      <<<BlocksNum4ThreadsNum(y_shape.Count(0)), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          reinterpret_cast<const half*>(src), reinterpret_cast<half*>(dest), index_helper,
          y_shape.Count(0), y_shape.At(c_idx), y_shape.At(d_idx), y_shape.At(h_idx),
          y_shape.At(w_idx), x_shape.At(d_idx), x_shape.At(h_idx), x_shape.At(w_idx), padding[4],
          padding[0], padding[2], static_cast<const half>(constant_value));
}

template<typename IN_T>
struct ConstantPad1dGradFunctor<DeviceType::kGPU, IN_T> final {
  void operator()(DeviceCtx* ctx, const IN_T* src, IN_T* dest,
                  const NdIndexOffsetHelper<int64_t, 3>& index_helper, const ShapeView& dy_shape,
                  const ShapeView& dx_shape, const std::vector<int64_t>& padding) {
    const int64_t c_idx = 1;
    const int64_t w_idx = 2;
    DoCUDAConstantPad1dGrad<IN_T>
        <<<BlocksNum4ThreadsNum(dy_shape.Count(0)), kCudaThreadsNumPerBlock, 0,
           ctx->cuda_stream()>>>(src, dest, index_helper, dy_shape.Count(0), dy_shape.At(c_idx),
                                 dy_shape.At(w_idx), dx_shape.At(w_idx), padding[0]);
  }
};

// float16 implementation
template<>
void ConstantPad1dGradFunctor<DeviceType::kGPU, float16>::operator()(
    DeviceCtx* ctx, const float16* src, float16* dest,
    const NdIndexOffsetHelper<int64_t, 3>& index_helper, const ShapeView& dy_shape,
    const ShapeView& dx_shape, const std::vector<int64_t>& padding) {
  const int64_t c_idx = 1;
  const int64_t w_idx = 2;
  DoCUDAConstantPad1dGrad<half>
      <<<BlocksNum4ThreadsNum(dy_shape.Count(0)), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          reinterpret_cast<const half*>(src), reinterpret_cast<half*>(dest), index_helper,
          dy_shape.Count(0), dy_shape.At(c_idx), dy_shape.At(w_idx), dx_shape.At(w_idx),
          padding[0]);
}

template<typename IN_T>
struct ConstantPad3dGradFunctor<DeviceType::kGPU, IN_T> final {
  void operator()(DeviceCtx* ctx, const IN_T* src, IN_T* dest,
                  const NdIndexOffsetHelper<int64_t, 5>& index_helper, const ShapeView& dy_shape,
                  const ShapeView& dx_shape, const std::vector<int64_t>& padding) {
    const int64_t c_idx = 1;
    const int64_t d_idx = 2;
    const int64_t h_idx = 3;
    const int64_t w_idx = 4;
    DoCUDAConstantPad3dGrad<IN_T><<<BlocksNum4ThreadsNum(dy_shape.Count(0)),
                                    kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        src, dest, index_helper, dy_shape.Count(0), dy_shape.At(c_idx), dy_shape.At(d_idx),
        dy_shape.At(h_idx), dy_shape.At(w_idx), dx_shape.At(d_idx), dx_shape.At(h_idx),
        dx_shape.At(w_idx), padding[4], padding[0], padding[2]);
  }
};

// float16 implementation
template<>
void ConstantPad3dGradFunctor<DeviceType::kGPU, float16>::operator()(
    DeviceCtx* ctx, const float16* src, float16* dest,
    const NdIndexOffsetHelper<int64_t, 5>& index_helper, const ShapeView& dy_shape,
    const ShapeView& dx_shape, const std::vector<int64_t>& padding) {
  const int64_t c_idx = 1;
  const int64_t d_idx = 2;
  const int64_t h_idx = 3;
  const int64_t w_idx = 4;
  DoCUDAConstantPad3dGrad<half>
      <<<BlocksNum4ThreadsNum(dy_shape.Count(0)), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          reinterpret_cast<const half*>(src), reinterpret_cast<half*>(dest), index_helper,
          dy_shape.Count(0), dy_shape.At(c_idx), dy_shape.At(d_idx), dy_shape.At(h_idx),
          dy_shape.At(w_idx), dx_shape.At(d_idx), dx_shape.At(h_idx), dx_shape.At(w_idx),
          padding[4], padding[0], padding[2]);
}

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_CONSTANT_PAD_FUNCTOR,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kGPU), PADDING_DATA_TYPE_GPU_SEQ);

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_CONSTANT_PAD_GRAD_FUNCTOR,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kGPU), PADDING_DATA_TYPE_GPU_SEQ);

}  // namespace user_op
}  // namespace oneflow
