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
#ifdef WITH_CUDA
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/reflection_pad_kernels_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {
namespace user_op {

template<typename IN_T>
__global__ void DoCUDAReflectionPad1d(const IN_T* src, IN_T* dest,
                                      const NdIndexOffsetHelper<int64_t, 3> index_helper,
                                      const int64_t elem_num, const int64_t src_num,
                                      const int64_t dest_num, const int64_t y_width,
                                      const int64_t x_width, const int64_t pad_left) {
  DoReflectionPad1d<IN_T>(src, dest, index_helper, elem_num, src_num, dest_num, y_width, x_width,
                          pad_left);
};

template<typename IN_T>
__global__ void DoCUDAReflectionPad1dGrad(const IN_T* src, IN_T* dest,
                                          const NdIndexOffsetHelper<int64_t, 3> index_helper,
                                          const int64_t elem_num, const int64_t src_num,
                                          const int64_t dest_num, const int64_t dy_width,
                                          const int64_t dx_width, const int64_t pad_left) {
  DoReflectionPad1dGrad<IN_T>(src, dest, index_helper, elem_num, src_num, dest_num, dy_width,
                              dx_width, pad_left);
};

template<typename IN_T>
__global__ void DoCUDAReflectionPad2d(const IN_T* src, IN_T* dest,
                                      const NdIndexOffsetHelper<int64_t, 4> index_helper,
                                      const int64_t elem_num, const int64_t src_num,
                                      const int64_t dest_num, const int64_t y_height,
                                      const int64_t y_width, const int64_t x_height,
                                      const int64_t x_width, const int64_t pad_left,
                                      const int64_t pad_top) {
  DoReflectionPad2d<IN_T>(src, dest, index_helper, elem_num, src_num, dest_num, y_height, y_width,
                          x_height, x_width, pad_left, pad_top);
};

template<typename IN_T>
__global__ void DoCUDAReflectionPad2dGrad(const IN_T* src, IN_T* dest,
                                          const NdIndexOffsetHelper<int64_t, 4> index_helper,
                                          const int64_t elem_num, const int64_t src_num,
                                          const int64_t dest_num, const int64_t dy_height,
                                          const int64_t dy_width, const int64_t dx_height,
                                          const int64_t dx_width, const int64_t pad_left,
                                          const int64_t pad_top) {
  DoReflectionPad2dGrad<IN_T>(src, dest, index_helper, elem_num, src_num, dest_num, dy_height,
                              dy_width, dx_height, dx_width, pad_left, pad_top);
};

template<typename IN_T>
struct ReflectionPad1dFunctor<DeviceType::kCUDA, IN_T> final {
  void operator()(ep::Stream* stream, const IN_T* src, IN_T* dest,
                  const NdIndexOffsetHelper<int64_t, 3>& index_helper, const int64_t n_batch,
                  const int64_t n_channel, const int64_t y_width, const int64_t x_width,
                  const int64_t pad_left) {
    const int64_t dest_num = n_channel * y_width;
    const int64_t src_num = n_channel * x_width;
    const int64_t elem_num = n_batch * dest_num;
    DoCUDAReflectionPad1d<IN_T><<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0,
                                  stream->As<ep::CudaStream>()->cuda_stream()>>>(
        src, dest, index_helper, elem_num, src_num, dest_num, y_width, x_width, pad_left);
  }
};

// float16 implementation
template<>
void ReflectionPad1dFunctor<DeviceType::kCUDA, float16>::operator()(
    ep::Stream* stream, const float16* src, float16* dest,
    const NdIndexOffsetHelper<int64_t, 3>& index_helper, const int64_t n_batch,
    const int64_t n_channel, const int64_t y_width, const int64_t x_width, const int64_t pad_left) {
  const int64_t dest_num = n_channel * y_width;
  const int64_t src_num = n_channel * x_width;
  const int64_t elem_num = n_batch * dest_num;
  DoCUDAReflectionPad1d<half><<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0,
                                stream->As<ep::CudaStream>()->cuda_stream()>>>(
      reinterpret_cast<const half*>(src), reinterpret_cast<half*>(dest), index_helper, elem_num,
      src_num, dest_num, y_width, x_width, pad_left);
}

template<typename IN_T>
struct ReflectionPad1dGradFunctor<DeviceType::kCUDA, IN_T> final {
  void operator()(ep::Stream* stream, const IN_T* src, IN_T* dest,
                  const NdIndexOffsetHelper<int64_t, 3>& index_helper, const int64_t n_batch,
                  const int64_t n_channel, const int64_t dy_width, const int64_t dx_width,
                  const int64_t pad_left) {
    const int64_t dest_num = n_channel * dx_width;
    const int64_t src_num = n_channel * dy_width;
    const int64_t elem_num = n_batch * src_num;
    DoCUDAReflectionPad1dGrad<IN_T><<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0,
                                      stream->As<ep::CudaStream>()->cuda_stream()>>>(
        src, dest, index_helper, elem_num, src_num, dest_num, dy_width, dx_width, pad_left);
  }
};

// float16 implementation
template<>
void ReflectionPad1dGradFunctor<DeviceType::kCUDA, float16>::operator()(
    ep::Stream* stream, const float16* src, float16* dest,
    const NdIndexOffsetHelper<int64_t, 3>& index_helper, const int64_t n_batch,
    const int64_t n_channel, const int64_t dy_width, const int64_t dx_width,
    const int64_t pad_left) {
  const int64_t dest_num = n_channel * dx_width;
  const int64_t src_num = n_channel * dy_width;
  const int64_t elem_num = n_batch * src_num;
  DoCUDAReflectionPad1dGrad<half><<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0,
                                    stream->As<ep::CudaStream>()->cuda_stream()>>>(
      reinterpret_cast<const half*>(src), reinterpret_cast<half*>(dest), index_helper, elem_num,
      src_num, dest_num, dy_width, dx_width, pad_left);
}

template<typename IN_T>
struct ReflectionPad2dFunctor<DeviceType::kCUDA, IN_T> final {
  void operator()(ep::Stream* stream, const IN_T* src, IN_T* dest,
                  const NdIndexOffsetHelper<int64_t, 4>& index_helper, const int64_t n_batch,
                  const int64_t n_channel, const int64_t y_height, const int64_t y_width,
                  const int64_t x_height, const int64_t x_width, const int64_t pad_left,
                  const int64_t pad_top) {
    const int64_t dest_num = n_channel * y_height * y_width;
    const int64_t src_num = n_channel * x_height * x_width;
    const int64_t elem_num = n_batch * dest_num;
    DoCUDAReflectionPad2d<IN_T><<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0,
                                  stream->As<ep::CudaStream>()->cuda_stream()>>>(
        src, dest, index_helper, elem_num, src_num, dest_num, y_height, y_width, x_height, x_width,
        pad_left, pad_top);
  }
};

// float16 implementation
template<>
void ReflectionPad2dFunctor<DeviceType::kCUDA, float16>::operator()(
    ep::Stream* stream, const float16* src, float16* dest,
    const NdIndexOffsetHelper<int64_t, 4>& index_helper, const int64_t n_batch,
    const int64_t n_channel, const int64_t y_height, const int64_t y_width, const int64_t x_height,
    const int64_t x_width, const int64_t pad_left, const int64_t pad_top) {
  const int64_t dest_num = n_channel * y_height * y_width;
  const int64_t src_num = n_channel * x_height * x_width;
  const int64_t elem_num = n_batch * dest_num;
  DoCUDAReflectionPad2d<half><<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0,
                                stream->As<ep::CudaStream>()->cuda_stream()>>>(
      reinterpret_cast<const half*>(src), reinterpret_cast<half*>(dest), index_helper, elem_num,
      src_num, dest_num, y_height, y_width, x_height, x_width, pad_left, pad_top);
}

template<typename IN_T>
struct ReflectionPad2dGradFunctor<DeviceType::kCUDA, IN_T> final {
  void operator()(ep::Stream* stream, const IN_T* src, IN_T* dest,
                  const NdIndexOffsetHelper<int64_t, 4>& index_helper, const int64_t n_batch,
                  const int64_t n_channel, const int64_t dy_height, const int64_t dy_width,
                  const int64_t dx_height, const int64_t dx_width, const int64_t pad_left,
                  const int64_t pad_top) {
    const int64_t dest_num = n_channel * dx_height * dx_width;
    const int64_t src_num = n_channel * dy_height * dy_width;
    const int64_t elem_num = n_batch * src_num;
    DoCUDAReflectionPad2dGrad<IN_T><<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0,
                                      stream->As<ep::CudaStream>()->cuda_stream()>>>(
        src, dest, index_helper, elem_num, src_num, dest_num, dy_height, dy_width, dx_height,
        dx_width, pad_left, pad_top);
  }
};

// float16 implementation
template<>
void ReflectionPad2dGradFunctor<DeviceType::kCUDA, float16>::operator()(
    ep::Stream* stream, const float16* src, float16* dest,
    const NdIndexOffsetHelper<int64_t, 4>& index_helper, const int64_t n_batch,
    const int64_t n_channel, const int64_t dy_height, const int64_t dy_width,
    const int64_t dx_height, const int64_t dx_width, const int64_t pad_left,
    const int64_t pad_top) {
  const int64_t dest_num = n_channel * dx_height * dx_width;
  const int64_t src_num = n_channel * dy_height * dy_width;
  const int64_t elem_num = n_batch * src_num;
  DoCUDAReflectionPad2dGrad<half><<<BlocksNum4ThreadsNum(elem_num), kCudaThreadsNumPerBlock, 0,
                                    stream->As<ep::CudaStream>()->cuda_stream()>>>(
      reinterpret_cast<const half*>(src), reinterpret_cast<half*>(dest), index_helper, elem_num,
      src_num, dest_num, dy_height, dy_width, dx_height, dx_width, pad_left, pad_top);
}

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_REFLECTION_PAD_FUNCTOR,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCUDA),
                                 PADDING_DATA_TYPE_CUDA_SEQ);

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_REFLECTION_PAD_GRAD_FUNCTOR,
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCUDA),
                                 PADDING_DATA_TYPE_CUDA_SEQ);

}  // namespace user_op
}  // namespace oneflow

#endif  // WITH_CUDA
