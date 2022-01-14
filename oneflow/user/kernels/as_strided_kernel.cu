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
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/consistency_check.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace {

constexpr size_t NUM_DIM = 8;

template<typename T>
__global__ void AsStrided_kernel(const T* input_buf, T* output_buf, const int64_t* dest_dims, const int32_t* stride,
                const int32_t dest_num_dims, const int32_t storage_offset, const int32_t input_num, const int32_t output_num) {
  NdIndexOffsetHelper<int64_t,NUM_DIM> destIndexOffsetHelper(dest_dims, dest_num_dims);
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, output_num) {
     int64_t dst_index[NUM_DIM];
     destIndexOffsetHelper.OffsetToNdIndex(i, dst_index, dest_num_dims);
     int32_t index_in_input = storage_offset;
     FOR_RANGE(int64_t, j, 0, dest_num_dims){
        index_in_input += dst_index[0];
        //index_in_input+=dst_index[j]*stride[j];
     }
     output_buf[i] = input_buf[index_in_input];
  }
}

template<typename T>
__global__ void AsStridedGrad_kernel(const T* dy_buf, T* dx_buf, const int64_t* dy_dims, const int32_t* stride,
                const int32_t dy_num_dims, const int32_t storage_offset, const int32_t dx_num, const int32_t dy_num) {
    NdIndexOffsetHelper<int64_t,NUM_DIM> destIndexOffsetHelper(dy_dims, dy_num_dims);
    CUDA_1D_KERNEL_LOOP_T(int64_t, i, dy_num) {
        int64_t dy_index[NUM_DIM];
        destIndexOffsetHelper.OffsetToNdIndex(i, dy_index,dy_num_dims);
        int32_t index_in_dx = storage_offset;
        FOR_RANGE(int64_t, j, 0, dy_num_dims){
            index_in_dx+=dy_index[j]*stride[j];
        }
        dx_buf[index_in_dx] += dy_buf[i];
    }
}

template<typename T>
struct AsStridedFunctor final {
void operator()(ep::Stream* stream, const T* input_buf, T* output_buf, const int64_t* dest_dims, const int32_t* stride,
                const int32_t dest_num_dims, const int32_t storage_offset, const int32_t input_num, const int32_t output_num) {
    AsStrided_kernel<T><<<BlocksNum4ThreadsNum(output_num), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(input_buf, output_buf, dest_dims, stride,
           dest_num_dims, storage_offset, input_num, output_num);
}
};

template<typename T>
struct AsStridedGradFunctor final {
 void operator()(ep::Stream* stream, const T* dy_buf, T* dx_buf, const int64_t* dy_dims, const int32_t* stride,
                const int32_t dy_num_dims, const int32_t storage_offset, const int32_t dx_num, const int32_t dy_num) {
    AsStridedGrad_kernel<T><<<BlocksNum4ThreadsNum(dy_num), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(dy_buf, dx_buf, dy_dims, stride,
                dy_num_dims, storage_offset, dx_num, dy_num);
}
};

}

template<typename T>
class GpuAsStridedKernel final : public user_op::OpKernel {
 public:
  GpuAsStridedKernel() = default;
  ~GpuAsStridedKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("output", 0);
    const auto size = ctx->Attr<std::vector<int32_t>>("size");
    const auto stride = ctx->Attr<std::vector<int32_t>>("stride");
    const int32_t storage_offset = ctx->Attr<int32_t>("storage_offset");
    
    size_t dest_num_dims = output->shape().NumAxes();
    const int64_t *dest_dims = output->shape().ptr();
    const size_t input_num = input->shape().Count(0);
    const size_t output_num = output->shape().Count(0);

    AsStridedFunctor<T>()(ctx->stream(), input->dptr<T>(), output->mut_dptr<T>(), dest_dims, stride.data(), dest_num_dims, storage_offset,
                          input_num, output_num);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class GpuAsStridedGradKernel final : public user_op::OpKernel {
 public:
  GpuAsStridedGradKernel() = default;
  ~GpuAsStridedGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const auto size = ctx->Attr<std::vector<int32_t>>("size");
    const auto stride = ctx->Attr<std::vector<int32_t>>("stride");
    const int32_t storage_offset = ctx->Attr<int32_t>("storage_offset");

    size_t dy_num_dims = dy->shape().NumAxes();
    const int64_t *dy_dims = dy->shape().ptr();
    const size_t dx_num = dx->shape().Count(0);
    const size_t dy_num = dy->shape().Count(0);
    
    Memset<DeviceType::kCPU>(ctx->stream(), dx->mut_dptr(), 0, dx->shape().Count(0) * sizeof(T));
    
    AsStridedGradFunctor<T>()(ctx->stream(), dy->dptr<T>(), dx->mut_dptr<T>(), dy_dims, stride.data(), dy_num_dims, storage_offset,
                          dx_num, dy_num);

  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};



#define REGISTER_GPUASSTRIDED_KERNEL(in_type)                                     \
    REGISTER_USER_KERNEL("as_strided")                         \
      .SetCreateFn<                                            \
          GpuAsStridedKernel<in_type>>()                       \
      .SetIsMatchedHob(                                        \
          (user_op::HobDeviceType() == DeviceType::kCUDA)       \
          && (user_op::HobDataType("input", 0) == GetDataType<in_type>::value));   \
    REGISTER_USER_KERNEL("as_strided_grad")                    \
      .SetCreateFn<                                            \
          GpuAsStridedGradKernel<in_type>>()                   \
      .SetIsMatchedHob(                                        \
          (user_op::HobDeviceType() == DeviceType::kCUDA)       \
          && (user_op::HobDataType("input", 0) == GetDataType<in_type>::value));            

REGISTER_GPUASSTRIDED_KERNEL(float);
REGISTER_GPUASSTRIDED_KERNEL(double);
REGISTER_GPUASSTRIDED_KERNEL(int8_t);
REGISTER_GPUASSTRIDED_KERNEL(int32_t);
REGISTER_GPUASSTRIDED_KERNEL(int64_t);

#undef  REGISTER_GPUASSTRIDED_KERNEL

}  // namespace oneflow