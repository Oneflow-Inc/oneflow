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
#include "oneflow/core/cuda/atomic.cuh"
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

template<size_t num_dims, typename IndexType>
struct SelectParams {
  NdIndexOffsetHelper<IndexType, num_dims> inputIndexOffsetHelper;
  NdIndexOffsetHelper<IndexType, num_dims> outputIndexOffsetHelper;
  int64_t input_dims[num_dims];
  int32_t output_dims[num_dims];
  int32_t dim;
  int32_t index;
  int32_t input_dims_num;
  int32_t output_dims_num;
  int32_t output_num;
};

template<typename T>
__global__ void Select_kernel(const T* input_buf, T* output_buf, SelectParams<NUM_DIM, int64_t> params) {
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, params.output_num) {
     int64_t src_index[NUM_DIM];
     int64_t dst_index[NUM_DIM];
     params.outputIndexOffsetHelper.OffsetToNdIndex(i, dst_index, params.output_dims_num);
     FOR_RANGE(int64_t, j, 0, params.input_dims_num){
        if(j<params.dim){
            src_index[j] = dst_index[j];
        }else if(j==params.dim){
            src_index[j] = params.index;
        }else{
            src_index[j] = dst_index[j-1];
        }
    }
    int64_t index_in_input = params.inputIndexOffsetHelper.NdIndexToOffset(src_index, params.input_dims_num);
    output_buf[i] = input_buf[index_in_input];
  }

}

template<typename T>
__global__ void SelectGrad_kernel(const T* dy_buf, T* dx_buf, SelectParams<NUM_DIM, int64_t> params) {
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, params.output_num) {
     int64_t src_index[NUM_DIM];
     int64_t dst_index[NUM_DIM];
     params.outputIndexOffsetHelper.OffsetToNdIndex(i, dst_index, params.output_dims_num);
     FOR_RANGE(int64_t, j, 0, params.input_dims_num){
        if(j<params.dim){
            src_index[j] = dst_index[j];
        }else if(j==params.dim){
            src_index[j] = params.index;
        }else{
            src_index[j] = dst_index[j-1];
        }
    }
    int64_t index_in_input = params.inputIndexOffsetHelper.NdIndexToOffset(src_index, params.input_dims_num);
    dx_buf[index_in_input] = dy_buf[i];
  }

}

template<typename T>
struct SelectFunctor final {
Maybe<void> operator()(ep::Stream* stream, const T* input_buf, T* output_buf, const int32_t dim, const int32_t index,
                      const int64_t* input_dims, const int64_t* output_dims, 
                      const int32_t input_dims_num, const int32_t output_dims_num, const int32_t output_num) {  
    NdIndexOffsetHelper<int64_t, NUM_DIM> inputIndexOffsetHelper(input_dims, input_dims_num);
    NdIndexOffsetHelper<int64_t, NUM_DIM> outputIndexOffsetHelper(output_dims, output_dims_num);
    SelectParams<NUM_DIM, int64_t> params;
    params.inputIndexOffsetHelper = inputIndexOffsetHelper;
    params.outputIndexOffsetHelper = outputIndexOffsetHelper;
    params.dim = dim;
    params.index = index;
    params.input_dims_num = input_dims_num;
    params.output_dims_num = output_dims_num;
    params.output_num = output_num;
    FOR_RANGE(size_t, i, 0, input_dims_num){
      params.input_dims[i] = input_dims[i];
      if(i<output_dims_num){
         params.output_dims[i] = output_dims[i];
      }
    }
    
    Select_kernel<T><<<BlocksNum4ThreadsNum(output_num), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(input_buf, output_buf, params);

    return Maybe<void>::Ok();
}
};

template<typename T>
struct SelectGradFunctor final {
Maybe<void> operator()(ep::Stream* stream, const T* dy_buf, T* dx_buf, const int32_t dim, const int32_t index,
                      const int64_t* dx_dims, const int64_t* dy_dims, 
                      const int32_t dx_dims_num, const int32_t dy_dims_num, const int32_t dy_num) {

    NdIndexOffsetHelper<int64_t, NUM_DIM> dxIndexOffsetHelper(dx_dims, dx_dims_num);
    NdIndexOffsetHelper<int64_t, NUM_DIM> dyIndexOffsetHelper(dy_dims, dy_dims_num);
    SelectParams<NUM_DIM, int64_t> params;
    params.inputIndexOffsetHelper = dxIndexOffsetHelper;
    params.outputIndexOffsetHelper = dyIndexOffsetHelper;
    params.dim = dim;
    params.index = index;
    params.input_dims_num = dx_dims_num;
    params.output_dims_num = dy_dims_num;
    params.output_num = dy_num;
    FOR_RANGE(size_t, i, 0, dx_dims_num){
      params.input_dims[i] = dx_dims[i];
      if(i<dy_dims_num){
         params.output_dims[i] = dy_dims[i];
      }
    }
    
    SelectGrad_kernel<T><<<BlocksNum4ThreadsNum(dy_num), kCudaThreadsNumPerBlock, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(dy_buf, dx_buf, params);
    return Maybe<void>::Ok();
}
};

}

template<typename T>
class GpuSelectKernel final : public user_op::OpKernel {
 public:
  GpuSelectKernel() = default;
  ~GpuSelectKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("output", 0);
    const int32_t dim = ctx->Attr<int32_t>("dim");
    const int32_t index = ctx->Attr<int32_t>("index");
    
    size_t input_dims_num = input->shape().NumAxes();
    size_t output_dims_num = output->shape().NumAxes();
    const int64_t *input_dims = input->shape().ptr();
    const int64_t *output_dims = output->shape().ptr();
    const size_t output_num = output->shape().Count(0);

    SelectFunctor<T>()(ctx->stream(), input->dptr<T>(), output->mut_dptr<T>(), dim, index, 
                       input_dims, output_dims, input_dims_num, output_dims_num, output_num);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class GpuSelectGradKernel final : public user_op::OpKernel {
 public:
  GpuSelectGradKernel() = default;
  ~GpuSelectGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int32_t dim = ctx->Attr<int32_t>("dim");
    const int32_t index = ctx->Attr<int32_t>("index");

    size_t dx_dims_num = dx->shape().NumAxes();
    size_t dy_dims_num = dy->shape().NumAxes();
    const int64_t *dx_dims = dx->shape().ptr();
    const int64_t *dy_dims = dy->shape().ptr();
    const size_t dy_num = dy->shape().Count(0);
    
    Memset<DeviceType::kCUDA>(ctx->stream(), dx->mut_dptr(), 0, dx->shape().Count(0) * sizeof(T));
    SelectGradFunctor<T>()(ctx->stream(), dy->dptr<T>(), dx->mut_dptr<T>(), dim, index, 
                       dx_dims, dy_dims, dx_dims_num, dy_dims_num, dy_num);

  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};



#define REGISTER_GPUSELECT_KERNEL(in_type)                                     \
    REGISTER_USER_KERNEL("select")                         \
      .SetCreateFn<                                            \
          GpuSelectKernel<in_type>>()                       \
      .SetIsMatchedHob(                                        \
          (user_op::HobDeviceType() == DeviceType::kCUDA)       \
          && (user_op::HobDataType("input", 0) == GetDataType<in_type>::value));   \
    REGISTER_USER_KERNEL("select_grad")                    \
      .SetCreateFn<                                            \
          GpuSelectGradKernel<in_type>>()                   \
      .SetIsMatchedHob(                                        \
          (user_op::HobDeviceType() == DeviceType::kCUDA)       \
          && (user_op::HobDataType("input", 0) == GetDataType<in_type>::value));            

REGISTER_GPUSELECT_KERNEL(float);
REGISTER_GPUSELECT_KERNEL(double);
REGISTER_GPUSELECT_KERNEL(half);

#undef  REGISTER_GPUSELECT_KERNEL

}  // namespace oneflow