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
#include "oneflow/core/device/memory_copier.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/user/kernels/reflection_pad2d_kernel_util.h"

namespace oneflow {
namespace user_op {

namespace {

// Fill ShapeView into dim vector
DimVector ShapeViewToDimVector(const ShapeView& tensor_shape) {
  int64_t ndims = tensor_shape.NumAxes();
  DimVector shape_vec(ndims);
  for (int64_t i = 0; i < ndims; ++i) { shape_vec[i] = tensor_shape.At(i); }
  shape_vec[ndims - 1] = shape_vec[ndims - 1];
  return shape_vec;
}

template<DeviceType device_type, typename T>
void ReflectionPading(
  const T* src, T * dest,
  int64_t n_batch, int64_t n_channel,int64_t y_height, int64_t y_width,
  int64_t x_height, int64_t x_width, int64_t pad_left, int64_t pad_top
){
  int64_t ip_x, ip_y;
  for(int64_t n = 0; n<n_batch; n++){
      for(int64_t c = 0; c<n_channel; c++){
        for(int64_t i = 0; i<y_height; i++){
          for(int64_t j = 0; j<y_width; j++){
            if(j < pad_left){
              ip_x = pad_left * 2 - j;
            }else if( j >= pad_left && j < x_width + pad_left){
              ip_x = j;
            }else{
              ip_x = (x_width + pad_left - 1) * 2 - j;
            }

            if(i<pad_top){
              ip_y = pad_top * 2 - i;
            }else if(i >= pad_top && i < x_height + pad_top){
              ip_y = i;
            }else{
              ip_y = (x_height + pad_top - 1) * 2 - i;
            }
            ip_x = ip_x - pad_left;
            ip_y = ip_y - pad_top;
            int64_t  dest_index = c * y_width * y_height + i * y_width + j;
            int64_t src_index =  c * x_width * x_height + ip_y * x_width + ip_x;
            dest[dest_index] = src[src_index];
          }
        }
      }
    }

}


template<DeviceType device_type, typename T>
void ReflectionGradPading(
  const T* src, T * dest, int64_t n_batch, int64_t n_channel,
  int64_t dy_height, int64_t dy_width, int64_t dx_height, int64_t dx_width, 
  int64_t pad_left, int64_t pad_top
){
  int64_t ip_x, ip_y;
  for(int64_t n = 0; n<n_batch; n++){
    for(int64_t c = 0; c<n_channel; c++){
      for(int64_t i = 0; i<dy_height; i++){
        for(int64_t j = 0; j<dy_width; j++){
          printf("n:%ld, c:%ld, h:%ld, w:%ld\n", n, c, i, j);
          if(j < pad_left){
            ip_x = pad_left * 2 - j;
          }else if( j >= pad_left && j < dx_width + pad_left){
            ip_x = j;
          }else{
            ip_x = (dx_width + pad_left - 1) * 2 - j;
          }
        
          if(i<pad_top){
            ip_y = pad_top * 2 - i;
          }else if(i >= pad_top && i < dx_height + pad_top){
            ip_y = i;
          }else{
            ip_y = (dx_height + pad_top - 1) * 2 - i;
          }
          ip_x = ip_x - pad_left;
          ip_y = ip_y - pad_top;
          int64_t src_index =  c * dy_width * dy_height + i * dy_width + j;
          int64_t dest_index = c * dx_width *dx_height + ip_y * dx_width + ip_x;
          dest[dest_index] += src[src_index];
        }
      }
    }
  }
}

}  // namespace

template<DeviceType device_type, typename T>
class ReflectionPad2dKernel final : public OpKernel {
 public:
  ReflectionPad2dKernel() = default;
  ~ReflectionPad2dKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const auto& padding = ctx->Attr<std::vector<int64_t>>("padding");
    const std::string data_format = ctx->Attr<std::string>("data_format");
    const int64_t ndims = x->shape().NumAxes();
    CHECK_EQ(padding.size(), ndims);
    int64_t c_idx, h_idx, w_idx;
    if (data_format == "NCHW") {
      c_idx = 1; 
      h_idx = 2;
      w_idx = 3;
    } else {
      h_idx = 1;
      w_idx = 2;
      c_idx = 3;
    }

    int64_t pad_left = padding[w_idx];
    int64_t pad_top = padding[h_idx];

    int64_t n_batch = y->shape().At(0);
    int64_t n_channel = y->shape().At(c_idx);
    int64_t y_height = y->shape().At(h_idx);
    int64_t y_width = y->shape().At(w_idx);
    int64_t x_height = x->shape().At(h_idx);
    int64_t x_width = x->shape().At(w_idx);

    T * dest = y->mut_dptr<T>();
    const T* src = x->dptr<T>();


    ReflectionPad2dFunctor<device_type, T>()(
        ctx->device_ctx(), src, dest, n_batch, n_channel, 
        y_height, y_width, x_height, x_width, pad_left, pad_top
    );

  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_REFLECTION_PAD2D_KERNELS(dev, dtype)   \
  REGISTER_USER_KERNEL("reflection_pad2d")              \
      .SetCreateFn<ReflectionPad2dKernel<dev, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == dev) \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

#ifdef WITH_CUDA
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kGPU, float)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kGPU, double)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kGPU, float16)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kGPU, int32_t)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kGPU, int64_t)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kGPU, int8_t)
#endif
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kCPU, float)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kCPU, double)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kCPU, int32_t)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kCPU, int64_t)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kCPU, int8_t)

template<DeviceType device_type, typename T>
class ReflectionPad2dGradKernel final : public OpKernel {
 public:
  ReflectionPad2dGradKernel() = default;
  ~ReflectionPad2dGradKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const auto& padding = ctx->Attr<std::vector<int64_t>>("padding");
    const std::string data_format = ctx->Attr<std::string>("data_format");
    const int64_t ndims = dy->shape().NumAxes();
    CHECK_EQ(padding.size(), ndims);

    int64_t c_idx, h_idx, w_idx;
    if (data_format == "NCHW") {
      c_idx = 1; 
      h_idx = 2;
      w_idx = 3;
    } else {
      h_idx = 1;
      w_idx = 2;
      c_idx = 3;
    }

  
    int64_t pad_left = padding[w_idx];
    int64_t pad_top = padding[h_idx];
    int64_t n_batch = dy->shape().At(0);
    int64_t n_channel = dy->shape().At(c_idx);
    int64_t dy_height = dy->shape().At(h_idx);
    int64_t dy_width = dy->shape().At(w_idx);
    int64_t dx_height = dx->shape().At(h_idx);
    int64_t dx_width = dx->shape().At(w_idx);

    const T* src = dy->dptr<T>();
    T * dest = dx->mut_dptr<T>();

    ReflectionPad2dGradFunctor<device_type, T>()(
      ctx->device_ctx(), src, dest, n_batch, n_channel, 
      dy_height, dy_width, dx_height, dx_width, pad_left, pad_top
    );

    


  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(dev, dtype)  \
  REGISTER_USER_KERNEL("reflection_pad2d_grad")             \
      .SetCreateFn<ReflectionPad2dGradKernel<dev, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == dev)     \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

#ifdef WITH_CUDA
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kGPU, float)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kGPU, double)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kGPU, float16)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kGPU, int32_t)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kGPU, int64_t)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kGPU, int8_t)
#endif
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kCPU, float)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kCPU, double)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kCPU, int32_t)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kCPU, int64_t)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kCPU, int8_t)

}  // namespace user_op
}  // namespace oneflow