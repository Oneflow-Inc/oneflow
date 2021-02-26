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
#ifdef WITH_CUDA
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/pad2d_kernels_util.h"

namespace oneflow {
namespace user_op {

template<typename IN_T>
__global__ void DoCUDAReflectionPad2d(const IN_T *src, IN_T *dest,
                                      const NdIndexOffsetHelper<int64_t, 4> index_helper,
                                      int64_t elem_num, int64_t src_num, int64_t dest_num,
                                      int64_t y_height, int64_t y_width, int64_t x_height,
                                      int64_t x_width, int64_t pad_left, int64_t pad_top) {
  DoReflectionPad2d<IN_T>(src, dest, index_helper, elem_num, src_num, dest_num, y_height, y_width,
                          x_height, x_width, pad_left, pad_top);
};

template<typename IN_T>
__global__ void DoCUDAReflectionPad2dGrad(const IN_T *src, IN_T *dest,
                                          const NdIndexOffsetHelper<int64_t, 4> index_helper,
                                          int64_t elem_num, int64_t src_num, int64_t dest_num,
                                          int64_t dy_height, int64_t dy_width, int64_t dx_height,
                                          int64_t dx_width, int64_t pad_left, int64_t pad_top) {
  DoReflectionPad2dGrad<IN_T>(src, dest, index_helper, elem_num, src_num, dest_num, dy_height,
                              dy_width, dx_height, dx_width, pad_left, pad_top);
};

template<typename IN_T>
struct ReflectionPad2dFunctor<DeviceType::kGPU, IN_T> final {
  void operator()(DeviceCtx *ctx, const IN_T *src, IN_T *dest,
                  const NdIndexOffsetHelper<int64_t, 4> &index_helper, int64_t n_batch,
                  int64_t n_channel, int64_t y_height, int64_t y_width, int64_t x_height,
                  int64_t x_width, int64_t pad_left, int64_t pad_top) {
    int64_t dest_num = n_channel * y_height * y_width;
    int64_t src_num = n_channel * x_height * x_width;
    int64_t elem_num = n_batch * dest_num;
    DoCUDAReflectionPad2d<IN_T>
        <<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            src, dest, index_helper, elem_num, src_num, dest_num, y_height, y_width, x_height,
            x_width, pad_left, pad_top);
  }
};

// float16 implementation
template<>
void ReflectionPad2dFunctor<DeviceType::kGPU, float16>::operator()(
    DeviceCtx *ctx, const float16 *src, float16 *dest,
    const NdIndexOffsetHelper<int64_t, 4> &index_helper, int64_t n_batch, int64_t n_channel,
    int64_t y_height, int64_t y_width, int64_t x_height, int64_t x_width, int64_t pad_left,
    int64_t pad_top) {
  int64_t dest_num = n_channel * y_height * y_width;
  int64_t src_num = n_channel * x_height * x_width;
  int64_t elem_num = n_batch * dest_num;
  DoCUDAReflectionPad2d<half>
      <<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          reinterpret_cast<const half *>(src), reinterpret_cast<half *>(dest), index_helper,
          elem_num, src_num, dest_num, y_height, y_width, x_height, x_width, pad_left, pad_top);
}

template<typename IN_T>
struct ReflectionPad2dGradFunctor<DeviceType::kGPU, IN_T> final {
  void operator()(DeviceCtx *ctx, const IN_T *src, IN_T *dest,
                  const NdIndexOffsetHelper<int64_t, 4> &index_helper, int64_t n_batch,
                  int64_t n_channel, int64_t dy_height, int64_t dy_width, int64_t dx_height,
                  int64_t dx_width, int64_t pad_left, int64_t pad_top) {
    int64_t dest_num = n_channel * dx_height * dx_width;
    int64_t src_num = n_channel * dy_height * dy_width;
    int64_t elem_num = n_batch * src_num;
    DoCUDAReflectionPad2dGrad<IN_T>
        <<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            src, dest, index_helper, elem_num, src_num, dest_num, dy_height, dy_width, dx_height,
            dx_width, pad_left, pad_top);
  }
};

// float16 implementation
template<>
void ReflectionPad2dGradFunctor<DeviceType::kGPU, float16>::operator()(
    DeviceCtx *ctx, const float16 *src, float16 *dest,
    const NdIndexOffsetHelper<int64_t, 4> &index_helper, int64_t n_batch, int64_t n_channel,
    int64_t dy_height, int64_t dy_width, int64_t dx_height, int64_t dx_width, int64_t pad_left,
    int64_t pad_top) {
  int64_t dest_num = n_channel * dx_height * dx_width;
  int64_t src_num = n_channel * dy_height * dy_width;
  int64_t elem_num = n_batch * src_num;
  DoCUDAReflectionPad2dGrad<half>
      <<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          reinterpret_cast<const half *>(src), reinterpret_cast<half *>(dest), index_helper,
          elem_num, src_num, dest_num, dy_height, dy_width, dx_height, dx_width, pad_left, pad_top);
}

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_REFLECTION_PAD2D_FUNCTOR,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kGPU), PADDING_DATA_TYPE_GPU_SEQ);

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_REFLECTION_PAD2D_GRAD_FUNCTOR,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kGPU), PADDING_DATA_TYPE_GPU_SEQ);

template<typename IN_T>
__global__ void DoCUDAReplicationPad2d(const IN_T *src, IN_T *dest,
                                       const NdIndexOffsetHelper<int64_t, 4> index_helper,
                                       int64_t elem_num, int64_t src_num, int64_t dest_num,
                                       int64_t y_height, int64_t y_width, int64_t x_height,
                                       int64_t x_width, int64_t pad_left, int64_t pad_top) {
  DoReplicationPad2d<IN_T>(src, dest, index_helper, elem_num, src_num, dest_num, y_height, y_width,
                           x_height, x_width, pad_left, pad_top);
};

template<typename IN_T>
__global__ void DoCUDAReplicationPad2dGrad(const IN_T *src, IN_T *dest,
                                           const NdIndexOffsetHelper<int64_t, 4> index_helper,
                                           int64_t elem_num, int64_t src_num, int64_t dest_num,
                                           int64_t dy_height, int64_t dy_width, int64_t dx_height,
                                           int64_t dx_width, int64_t pad_left, int64_t pad_top) {
  DoReplicationPad2dGrad<IN_T>(src, dest, index_helper, elem_num, src_num, dest_num, dy_height,
                               dy_width, dx_height, dx_width, pad_left, pad_top);
};

template<typename IN_T>
struct ReplicationPad2dFunctor<DeviceType::kGPU, IN_T> final {
  void operator()(DeviceCtx *ctx, const IN_T *src, IN_T *dest,
                  const NdIndexOffsetHelper<int64_t, 4> &index_helper, int64_t n_batch,
                  int64_t n_channel, int64_t y_height, int64_t y_width, int64_t x_height,
                  int64_t x_width, int64_t pad_left, int64_t pad_top) {
    int64_t dest_num = n_channel * y_height * y_width;
    int64_t src_num = n_channel * x_height * x_width;
    int64_t elem_num = n_batch * dest_num;
    DoCUDAReplicationPad2d<IN_T>
        <<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            src, dest, index_helper, elem_num, src_num, dest_num, y_height, y_width, x_height,
            x_width, pad_left, pad_top);
  }
};

// float16 implementation
template<>
void ReplicationPad2dFunctor<DeviceType::kGPU, float16>::operator()(
    DeviceCtx *ctx, const float16 *src, float16 *dest,
    const NdIndexOffsetHelper<int64_t, 4> &index_helper, int64_t n_batch, int64_t n_channel,
    int64_t y_height, int64_t y_width, int64_t x_height, int64_t x_width, int64_t pad_left,
    int64_t pad_top) {
  int64_t dest_num = n_channel * y_height * y_width;
  int64_t src_num = n_channel * x_height * x_width;
  int64_t elem_num = n_batch * dest_num;
  DoCUDAReplicationPad2d<half>
      <<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          reinterpret_cast<const half *>(src), reinterpret_cast<half *>(dest), index_helper,
          elem_num, src_num, dest_num, y_height, y_width, x_height, x_width, pad_left, pad_top);
}

template<typename IN_T>
struct ReplicationPad2dGradFunctor<DeviceType::kGPU, IN_T> final {
  void operator()(DeviceCtx *ctx, const IN_T *src, IN_T *dest,
                  const NdIndexOffsetHelper<int64_t, 4> &index_helper, int64_t n_batch,
                  int64_t n_channel, int64_t dy_height, int64_t dy_width, int64_t dx_height,
                  int64_t dx_width, int64_t pad_left, int64_t pad_top) {
    int64_t dest_num = n_channel * dx_height * dx_width;
    int64_t src_num = n_channel * dy_height * dy_width;
    int64_t elem_num = n_batch * src_num;
    DoCUDAReplicationPad2dGrad<IN_T>
        <<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            src, dest, index_helper, elem_num, src_num, dest_num, dy_height, dy_width, dx_height,
            dx_width, pad_left, pad_top);
  }
};

// float16 implementation
template<>
void ReplicationPad2dGradFunctor<DeviceType::kGPU, float16>::operator()(
    DeviceCtx *ctx, const float16 *src, float16 *dest,
    const NdIndexOffsetHelper<int64_t, 4> &index_helper, int64_t n_batch, int64_t n_channel,
    int64_t dy_height, int64_t dy_width, int64_t dx_height, int64_t dx_width, int64_t pad_left,
    int64_t pad_top) {
  int64_t dest_num = n_channel * dx_height * dx_width;
  int64_t src_num = n_channel * dy_height * dy_width;
  int64_t elem_num = n_batch * src_num;
  DoCUDAReplicationPad2dGrad<half>
      <<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          reinterpret_cast<const half *>(src), reinterpret_cast<half *>(dest), index_helper,
          elem_num, src_num, dest_num, dy_height, dy_width, dx_height, dx_width, pad_left, pad_top);
}

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_REPLICATION_PAD2D_FUNCTOR,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kGPU), PADDING_DATA_TYPE_GPU_SEQ);

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_REPLICATION_PAD2D_GRAD_FUNCTOR,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kGPU), PADDING_DATA_TYPE_GPU_SEQ);

}  // namespace user_op
}  // namespace oneflow

#endif  // WITH_CUDA
