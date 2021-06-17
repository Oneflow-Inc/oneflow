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
#include "oneflow/user/ops/nn_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
using Im2ColFunc = void (*)(const T* in_dptr, const ShapeView& in_shape,
                            const ShapeView& weight_shape, const ShapeView& out_shape,
                            const int32_t* strides, const int32_t* dilation_rate,
                            const int32_t* padding_before, T* col_buf);

template<typename T>
using Col2ImFunc = void (*)(const T* col_buf, const ShapeView& in_shape,
                            const ShapeView& weight_shape, const ShapeView& out_shape,
                            const int32_t* strides, const int32_t* dilation_rate,
                            const int32_t* padding_before, T* in_diff_ptr);

template<typename T>
using GemmFunc = void (*)(enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b, const int m,
                          const int n, const int k, const T alpha, const T* a, const T* b,
                          const T beta, T* c);

template<typename T>
T* GetImgMutDptr(user_op::Tensor* tensor, int64_t idx) {
  return tensor->mut_dptr<T>() + tensor->shape().Count(1) * idx;
}

template<typename T>
const T* GetImgDptr(const user_op::Tensor* tensor, int64_t idx) {
  return tensor->dptr<T>() + tensor->shape().Count(1) * idx;
}

size_t CalcElemNumOfColBuf(const ShapeView& out_shape, const ShapeView& weight_shape,
                           const int32_t idx_offset) {
  int64_t col_buf_elem_cnt = 1;
  int64_t ndims = out_shape.NumAxes() - 2;
  for (size_t i = 0; i != ndims + 1; ++i) { col_buf_elem_cnt *= weight_shape.At(i + 1); }
  for (size_t i = 0; i != ndims; ++i) { col_buf_elem_cnt *= out_shape.At(idx_offset + i); }
  return col_buf_elem_cnt;
}

template<typename T>
struct ConvOpKernelState final : public user_op::OpKernelState {
  Im2ColFunc<T> im2col_func_;
  Col2ImFunc<T> col2im_func_;
  GemmFunc<T> forward_func_;

  Shape in_5d_shape_;
  Shape out_5d_shape_;
  Shape weight_5d_shape_;

  std::vector<int32_t> strides_3d_;
  std::vector<int32_t> dilation_rate_3d_;
  std::vector<int32_t> padding_before_3d_;

  enum CBLAS_TRANSPOSE is_out_diff_need_trans_;
  int32_t idx_offset_;
  bool is_dynamic_;

  void Update(const ShapeView& x_shape, const ShapeView& out_shape) {
    auto Gen5DShape = [](const ShapeView& shape, int32_t idx_offset) -> Shape {
      DimVector ret_vec;
      shape.ToDimVector(&ret_vec);
      int32_t ndims = ret_vec.size() - 2;
      ret_vec.insert(ret_vec.begin() + idx_offset, 3 - ndims, 1);
      return Shape(ret_vec);
    };
    if (is_dynamic_) {
      Shape in_shape;
      in_5d_shape_ = Gen5DShape(x_shape, idx_offset_);
      out_5d_shape_ = Gen5DShape(out_shape, idx_offset_);
    }
  }
};

template<typename T>
class DeconvCpuKernel final : public user_op::OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeconvCpuKernel);
  DeconvCpuKernel() = default;
  ~DeconvCpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* conv_state = dynamic_cast<ConvOpKernelState<T>*>(state);
    CHECK_NOTNULL(conv_state);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* col_buf = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    conv_state->Update(out->shape(), in->shape());
    Memset<DeviceType::kCPU>(ctx->device_ctx(), out->mut_dptr<T>(), 0,
                             out->shape().elem_cnt() * sizeof(T));

    int32_t idx_offset = conv_state->idx_offset_;
    FOR_RANGE(int64_t, i, 0, in->shape().At(0)) {
      // channels first:  col_buf' = weight(T) * out[i]'
      // channels last :  col_buf' = weight(T) * out[i]'(T)
      NewKernelUtil<DeviceType::kCPU>::OFGemm(
          nullptr, CblasTrans, conv_state->is_out_diff_need_trans_,
          conv_state->weight_5d_shape_.Count(1),                        //  ci * kd * kh * kw
          conv_state->out_5d_shape_.Count(idx_offset, idx_offset + 3),  //  od * oh * ow
          conv_state->weight_5d_shape_.At(0),                           //  weight
          static_cast<T>(1), weight->dptr<T>(), GetImgDptr<T>(in, i), static_cast<T>(0),
          col_buf->mut_dptr<T>());

      // in' = col2im(col_buf')
      conv_state->col2im_func_(col_buf->dptr<T>(), ShapeView(conv_state->in_5d_shape_),
                               ShapeView(conv_state->weight_5d_shape_),
                               ShapeView(conv_state->out_5d_shape_), conv_state->strides_3d_.data(),
                               conv_state->dilation_rate_3d_.data(),
                               conv_state->padding_before_3d_.data(), GetImgMutDptr<T>(out, i));
    }
  }
};

#define REGISTER_DECONV_DATA_GRAD_KERNEL(op_name, dtype)                                  \
  REGISTER_USER_KERNEL(#op_name)                                                          \
      .SetCreateFn<DeconvCpuKernel<dtype>>()                                              \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                                 \
                       & (user_op::HobAttr<int32_t>("groups") == 1)                       \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))    \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                       \
        size_t tmp_buffer_size = 0;                                                       \
        const auto& in_shape = ctx->TensorDesc4ArgNameAndIndex("in", 0)->shape();         \
        const auto& weight_shape = ctx->TensorDesc4ArgNameAndIndex("weight", 0)->shape(); \
                                                                                          \
        int64_t idx_offset = IdxOffset(ctx->Attr<std::string>("data_format"));            \
        tmp_buffer_size +=                                                                \
            CalcElemNumOfColBuf(in_shape, weight_shape, idx_offset) * sizeof(dtype);      \
        return tmp_buffer_size;                                                           \
      })

REGISTER_DECONV_DATA_GRAD_KERNEL(deconv1d, float);
REGISTER_DECONV_DATA_GRAD_KERNEL(deconv1d, double);
REGISTER_DECONV_DATA_GRAD_KERNEL(deconv2d, float);
REGISTER_DECONV_DATA_GRAD_KERNEL(deconv2d, double);
REGISTER_DECONV_DATA_GRAD_KERNEL(deconv3d, float);
REGISTER_DECONV_DATA_GRAD_KERNEL(deconv3d, double);

}  // namespace

}  // namespace oneflow
