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
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

#include "grid_sample_kernel_util.h"

namespace oneflow {

class CudnnGridSampleDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnGridSampleDesc);
  CudnnGridSampleDesc(DataType data_type, const ShapeView& shape) {
    std::vector<int> tensor_dim({shape.ptr(), shape.ptr() + shape.NumAxes()});
    OF_CUDNN_CHECK(cudnnCreateSpatialTransformerDescriptor(&val_));
    OF_CUDNN_CHECK(cudnnSetSpatialTransformerNdDescriptor(val_, CUDNN_SAMPLER_BILINEAR,
                                                          GetCudnnDataType(data_type),
                                                          shape.NumAxes(), tensor_dim.data()));
  }

  ~CudnnGridSampleDesc() { OF_CUDNN_CHECK(cudnnDestroySpatialTransformerDescriptor(val_)); }

  const cudnnSpatialTransformerDescriptor_t& Get() const { return val_; }

 private:
  cudnnSpatialTransformerDescriptor_t val_;
};

template<typename T>
struct CudnnGridSampleKernelUtil {
  static bool CanRunWithCudnn(user_op::KernelComputeContext* ctx) {
    if (ctx->Attr<std::string>("interpolation_mode") != "bilinear"
        || ctx->Attr<std::string>("padding_mode") != "zeros" || !ctx->Attr<bool>("align_corners")) {
      return false;
    }
    const ShapeView& input_shape = ctx->Tensor4ArgNameAndIndex("input", 0)->shape_view();
    if (input_shape.NumAxes() != 4 || input_shape.At(1) > 1024) { return false; }

    return true;
  }

  static void ForwardCompute(user_op::KernelComputeContext* ctx) {
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    const user_op::Tensor* grid = ctx->Tensor4ArgNameAndIndex("grid", 0);
    user_op::Tensor* output = ctx->Tensor4ArgNameAndIndex("output", 0);
    const ShapeView& input_shape = input->shape_view();
    const ShapeView& output_shape = output->shape_view();
    const DataType dtype = input->data_type();

    CudnnTensorDesc input_desc(dtype, input_shape, "channels_first");
    CudnnTensorDesc output_desc(dtype, output_shape, "channels_first");
    CudnnGridSampleDesc transfomer_desc(dtype, output_shape);

    OF_CUDNN_CHECK(cudnnSpatialTfSamplerForward(
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), transfomer_desc.Get(),
        CudnnSPOnePtr<T>(), input_desc.Get(), input->dptr(), grid->dptr(), CudnnSPZeroPtr<T>(),
        output_desc.Get(), output->mut_dptr()));
  }

  static void BackwardCompute(user_op::KernelComputeContext* ctx) {
    const user_op::Tensor* doutput = ctx->Tensor4ArgNameAndIndex("doutput", 0);
    const user_op::Tensor* input = ctx->Tensor4ArgNameAndIndex("input", 0);
    const user_op::Tensor* grid = ctx->Tensor4ArgNameAndIndex("grid", 0);
    user_op::Tensor* dinput = ctx->Tensor4ArgNameAndIndex("dinput", 0);
    user_op::Tensor* dgrid = ctx->Tensor4ArgNameAndIndex("dgrid", 0);
    const ShapeView& input_shape = input->shape_view();
    const ShapeView& output_shape = doutput->shape_view();
    const ShapeView& dinput_shape = dinput->shape_view();
    const DataType dtype = input->data_type();

    CudnnTensorDesc input_desc(dtype, input_shape, "channels_first");
    CudnnTensorDesc output_desc(dtype, output_shape, "channels_first");
    CudnnTensorDesc dinput_desc(dtype, dinput_shape, "channels_first");
    CudnnGridSampleDesc transfomer_desc(dtype, output_shape);

    OF_CUDNN_CHECK(cudnnSpatialTfSamplerBackward(
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), transfomer_desc.Get(),
        CudnnSPOnePtr<T>(), input_desc.Get(), input->dptr(), CudnnSPZeroPtr<T>(), dinput_desc.Get(),
        dinput->mut_dptr(), CudnnSPOnePtr<T>(), output_desc.Get(), doutput->dptr(), grid->dptr(),
        CudnnSPZeroPtr<T>(), dgrid->mut_dptr()));
  }
};

template<typename data_type, typename index_type>
__launch_bounds__(256) __global__
    void CUDAGridSampler4DKernel(const index_type nthreads, const data_type* input_ptr,
                                 const data_type* grid_ptr, data_type* output_ptr, index_type N,
                                 index_type C, index_type inp_H, index_type inp_W, index_type out_H,
                                 index_type out_W,
                                 const GridSamplerInterpolation interpolation_mode,
                                 const GridSamplerPadding padding_mode, const bool align_corners) {
  GridSampler4DKernel(nthreads, input_ptr, grid_ptr, output_ptr, N, C, inp_H, inp_W, out_H, out_W,
                      interpolation_mode, padding_mode, align_corners);
}

template<typename data_type, typename index_type>
__launch_bounds__(512) __global__
    void CUDAGridSampler5DKernel(const index_type nthreads, const data_type* input_ptr,
                                 const data_type* grid_ptr, data_type* output_ptr, index_type N,
                                 index_type C, index_type inp_D, index_type inp_H, index_type inp_W,
                                 index_type out_D, index_type out_H, index_type out_W,
                                 const GridSamplerInterpolation interpolation_mode,
                                 const GridSamplerPadding padding_mode, const bool align_corners) {
  GridSampler5DKernel(nthreads, input_ptr, grid_ptr, output_ptr, N, C, inp_D, inp_H, inp_W, out_D,
                      out_H, out_W, interpolation_mode, padding_mode, align_corners);
}

template<typename data_type, typename index_type>
__launch_bounds__(256) __global__ void CUDAGridSampler4DBackwardKernel(
    const index_type nthreads, const data_type* grad_output_ptr, const data_type* input_ptr,
    const data_type* grid_ptr, data_type* grad_input_ptr, data_type* grad_grid_ptr, index_type N,
    index_type C, index_type inp_H, index_type inp_W, index_type out_H, index_type out_W,
    const GridSamplerInterpolation interpolation_mode, const GridSamplerPadding padding_mode,
    const bool align_corners, const index_type grad_input_memory_span) {
  GridSampler4DBackwardKernel(nthreads, grad_output_ptr, input_ptr, grid_ptr, grad_input_ptr,
                              grad_grid_ptr, N, C, inp_H, inp_W, out_H, out_W, interpolation_mode,
                              padding_mode, align_corners, grad_input_memory_span);
}

template<typename data_type, typename index_type>
__launch_bounds__(256) __global__ void CUDAGridSampler5DBackwardKernel(
    const index_type nthreads, const data_type* grad_output_ptr, const data_type* input_ptr,
    const data_type* grid_ptr, data_type* grad_input_ptr, data_type* grad_grid_ptr, index_type N,
    index_type C, index_type inp_D, index_type inp_H, index_type inp_W, index_type out_D,
    index_type out_H, index_type out_W, const GridSamplerInterpolation interpolation_mode,
    const GridSamplerPadding padding_mode, const bool align_corners,
    const index_type grad_input_memory_span) {
  GridSampler5DBackwardKernel(nthreads, grad_output_ptr, input_ptr, grid_ptr, grad_input_ptr,
                              grad_grid_ptr, N, C, inp_D, inp_H, inp_W, out_D, out_H, out_W,
                              interpolation_mode, padding_mode, align_corners,
                              grad_input_memory_span);
}

template<typename data_type, typename index_type>
struct GridSampleKernelUtil<DeviceType::kCUDA, data_type, index_type> final {
  static void Forward4D(user_op::KernelComputeContext* ctx, const user_op::Tensor* input,
                        const user_op::Tensor* grid, user_op::Tensor* output,
                        GridSamplerInterpolation interpolation, GridSamplerPadding padding,
                        const bool align_corners, const ShapeView& input_shape,
                        const ShapeView& grid_shape, const ShapeView& output_shape, int64_t count) {
    if (CudnnGridSampleKernelUtil<data_type>::CanRunWithCudnn(ctx)
        && CanUse32BitIndex({input_shape, grid_shape, output_shape})) {
      return CudnnGridSampleKernelUtil<data_type>::ForwardCompute(ctx);
    }

    CUDAGridSampler4DKernel<data_type, index_type>
        <<<GridSampleGetBlocks(count, 256), 256, 0,
           ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
            count, input->dptr<data_type>(), grid->dptr<data_type>(), output->mut_dptr<data_type>(),
            input_shape.At(0), input_shape.At(1), input_shape.At(2), input_shape.At(3),
            output_shape.At(2), output_shape.At(3), interpolation, padding, align_corners);
  }
  static void Forward5D(user_op::KernelComputeContext* ctx, const user_op::Tensor* input,
                        const user_op::Tensor* grid, user_op::Tensor* output,
                        GridSamplerInterpolation interpolation, GridSamplerPadding padding,
                        const bool align_corners, const ShapeView& input_shape,
                        const ShapeView& grid_shape, const ShapeView& output_shape, int64_t count) {
    CUDAGridSampler5DKernel<data_type, index_type>
        <<<GridSampleGetBlocks(count, 512), 512, 0,
           ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
            count, input->dptr<data_type>(), grid->dptr<data_type>(), output->mut_dptr<data_type>(),
            input_shape.At(0), input_shape.At(1), input_shape.At(2), input_shape.At(3),
            input_shape.At(4), output_shape.At(2), output_shape.At(3), output_shape.At(4),
            interpolation, padding, align_corners);
  }

  static void Backward4D(user_op::KernelComputeContext* ctx, const user_op::Tensor* doutput,
                         const user_op::Tensor* input, const user_op::Tensor* grid,
                         user_op::Tensor* dinput, user_op::Tensor* dgrid,
                         GridSamplerInterpolation interpolation, GridSamplerPadding padding,
                         const bool align_corners, const ShapeView& input_shape,
                         const ShapeView& grid_shape, const ShapeView& output_shape,
                         int64_t count) {
    if (CudnnGridSampleKernelUtil<data_type>::CanRunWithCudnn(ctx)
        && CanUse32BitIndex({input_shape, grid_shape, output_shape})) {
      return CudnnGridSampleKernelUtil<data_type>::BackwardCompute(ctx);
    }

    CUDAGridSampler4DBackwardKernel<data_type, index_type>
        <<<GridSampleGetBlocks(count, 256), 256, 0,
           ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
            count, doutput->dptr<data_type>(), input->dptr<data_type>(), grid->dptr<data_type>(),
            dinput->mut_dptr<data_type>(), dgrid->mut_dptr<data_type>(), input_shape.At(0),
            input_shape.At(1), input_shape.At(2), input_shape.At(3), output_shape.At(2),
            output_shape.At(3), interpolation, padding, align_corners, input_shape.elem_cnt());
  }
  static void Backward5D(user_op::KernelComputeContext* ctx, const user_op::Tensor* doutput,
                         const user_op::Tensor* input, const user_op::Tensor* grid,
                         user_op::Tensor* dinput, user_op::Tensor* dgrid,
                         GridSamplerInterpolation interpolation, GridSamplerPadding padding,
                         const bool align_corners, const ShapeView& input_shape,
                         const ShapeView& grid_shape, const ShapeView& output_shape,
                         int64_t count) {
    CUDAGridSampler5DBackwardKernel<data_type, index_type>
        <<<GridSampleGetBlocks(count, 256), 256, 0,
           ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
            count, doutput->dptr<data_type>(), input->dptr<data_type>(), grid->dptr<data_type>(),
            dinput->mut_dptr<data_type>(), dgrid->mut_dptr<data_type>(), input_shape.At(0),
            input_shape.At(1), input_shape.At(2), input_shape.At(3), input_shape.At(4),
            output_shape.At(2), output_shape.At(3), output_shape.At(4), interpolation, padding,
            align_corners, input_shape.elem_cnt());
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_GRID_SAMPLE_KERNEL_UTIL, (DeviceType::kCUDA),
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);

}  // namespace oneflow
