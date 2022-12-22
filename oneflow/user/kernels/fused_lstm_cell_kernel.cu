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
#include <limits>
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/include/primitive/cast.h"
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/core/ep/cuda/cuda_device.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/user/kernels/fused_rnn_cell_kernel_util.h"

// NOTE(Liang Depeng): The implementation of fused_lstm_cell is modified from
//                     https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/RNN.cu

namespace oneflow {

namespace {

template<typename T>
struct AccumulateType {};
template<>
struct AccumulateType<float> {
  using type = float;
};
template<>
struct AccumulateType<double> {
  using type = double;
};

template<typename T>
using acc_type = typename AccumulateType<T>::type;

#define H2F(input) static_cast<ACC_T>(input)
#define F2H(input) static_cast<T>(input)

template<typename T>
__device__ __forceinline__ T sigmoid(T in) {
  T one = static_cast<T>(1.0);
  return one / (one + ::exp(-in));
}

template<typename T, typename ACC_T, typename IDX_TYPE>
#if __CUDA_ARCH__ >= 350
OF_LAUNCH_BOUNDS_2(512, 4)
#endif
__global__
    void lstm_cell_forward(const IDX_TYPE numel, const IDX_TYPE hidden_size,
                           const T* input_gates_ptr, const T* hidden_gates_ptr, const T* cx_ptr,
                           const T* input_bias_ptr, const T* hidden_bias_ptr, T* hy_ptr, T* cy_ptr,
                           T* workspace_ptr) {
  bool has_bias = input_bias_ptr != nullptr;
  for (IDX_TYPE linearIndex = blockIdx.x * blockDim.x + threadIdx.x; linearIndex < numel;
       linearIndex += gridDim.x * blockDim.x) {
    IDX_TYPE offset = (linearIndex / hidden_size) * 4 * hidden_size + linearIndex % hidden_size;

    T iig = input_gates_ptr[offset + 0 * hidden_size];
    T ifg = input_gates_ptr[offset + 1 * hidden_size];
    T icg = input_gates_ptr[offset + 2 * hidden_size];
    T iog = input_gates_ptr[offset + 3 * hidden_size];

    T hig = hidden_gates_ptr[offset + 0 * hidden_size];
    T hfg = hidden_gates_ptr[offset + 1 * hidden_size];
    T hcg = hidden_gates_ptr[offset + 2 * hidden_size];
    T hog = hidden_gates_ptr[offset + 3 * hidden_size];

    T* wig = &(workspace_ptr[offset + 0 * hidden_size]);
    T* wfg = &(workspace_ptr[offset + 1 * hidden_size]);
    T* wcg = &(workspace_ptr[offset + 2 * hidden_size]);
    T* wog = &(workspace_ptr[offset + 3 * hidden_size]);

    T cx = cx_ptr[linearIndex];

    T* hy = &(hy_ptr[linearIndex]);
    T* cy = &(cy_ptr[linearIndex]);

    T b1i, b1f, b1c, b1o;
    T b2i, b2f, b2c, b2o;

    if (has_bias) {
      b1i = input_bias_ptr[linearIndex % hidden_size + 0 * hidden_size];
      b1f = input_bias_ptr[linearIndex % hidden_size + 1 * hidden_size];
      b1c = input_bias_ptr[linearIndex % hidden_size + 2 * hidden_size];
      b1o = input_bias_ptr[linearIndex % hidden_size + 3 * hidden_size];

      b2i = hidden_bias_ptr[linearIndex % hidden_size + 0 * hidden_size];
      b2f = hidden_bias_ptr[linearIndex % hidden_size + 1 * hidden_size];
      b2c = hidden_bias_ptr[linearIndex % hidden_size + 2 * hidden_size];
      b2o = hidden_bias_ptr[linearIndex % hidden_size + 3 * hidden_size];
    } else {
      b1i = F2H(0.0);
      b1f = F2H(0.0);
      b1c = F2H(0.0);
      b1o = F2H(0.0);
      b2i = F2H(0.0);
      b2f = F2H(0.0);
      b2c = F2H(0.0);
      b2o = F2H(0.0);
    }

    ACC_T ig, fg, cg, og;
    ACC_T f_hy, f_cy;

    ig = sigmoid(H2F(iig) + H2F(hig) + H2F(b1i) + H2F(b2i));
    fg = sigmoid(H2F(ifg) + H2F(hfg) + H2F(b1f) + H2F(b2f));
    cg = ::tanh(H2F(icg) + H2F(hcg) + H2F(b1c) + H2F(b2c));
    og = sigmoid(H2F(iog) + H2F(hog) + H2F(b1o) + H2F(b2o));

    f_cy = (fg * H2F(cx)) + (ig * cg);
    f_hy = og * ::tanh(f_cy);

    *hy = F2H(f_hy);
    *cy = F2H(f_cy);

    // SAVE FOR BACKWARDS
    // Also need cy and cx but can be saved easily in python
    *wig = F2H(ig);
    *wfg = F2H(fg);
    *wcg = F2H(cg);
    *wog = F2H(og);
  }
}

template<typename T, typename ACC_T, typename IDX_TYPE>
#if __CUDA_ARCH__ >= 350
OF_LAUNCH_BOUNDS_2(512, 4)
#endif
__global__
    void lstm_cell_backward(const IDX_TYPE numel, const IDX_TYPE hidden_size, const T* grad_hy_ptr,
                            const T* grad_cy_ptr, const T* cx_ptr, const T* cy_ptr,
                            const T* workspace_ptr, T* grad_gates_ptr, T* grad_cx_ptr) {
  for (IDX_TYPE linearIndex = blockIdx.x * blockDim.x + threadIdx.x; linearIndex < numel;
       linearIndex += gridDim.x * blockDim.x) {
    IDX_TYPE offset = (linearIndex / hidden_size) * 4 * hidden_size + linearIndex % hidden_size;

    T ig = workspace_ptr[offset + 0 * hidden_size];
    T fg = workspace_ptr[offset + 1 * hidden_size];
    T cg = workspace_ptr[offset + 2 * hidden_size];
    T og = workspace_ptr[offset + 3 * hidden_size];

    T* ih = &(grad_gates_ptr[offset + 0 * hidden_size]);
    T* fh = &(grad_gates_ptr[offset + 1 * hidden_size]);
    T* ch = &(grad_gates_ptr[offset + 2 * hidden_size]);
    T* oh = &(grad_gates_ptr[offset + 3 * hidden_size]);

    // will return hidden grads here
    T cx = cx_ptr[linearIndex];
    T cy = cy_ptr[linearIndex];

    ACC_T go = H2F(grad_hy_ptr[linearIndex]);
    ACC_T goc = H2F(grad_cy_ptr[linearIndex]);

    ACC_T gcx = ::tanh(H2F(cy));

    ACC_T gog = go * gcx;
    gcx = go * H2F(og) * (1 - gcx * gcx) + goc;

    ACC_T gig = gcx * H2F(cg);
    ACC_T gfg = gcx * H2F(cx);
    ACC_T gcg = gcx * H2F(ig);

    gig = gig * (1 - H2F(ig)) * H2F(ig);
    gfg = gfg * (1 - H2F(fg)) * H2F(fg);
    gcg = gcg * (1 - H2F(cg) * H2F(cg));
    gog = gog * (1 - H2F(og)) * H2F(og);

    *ih = F2H(gig);
    *fh = F2H(gfg);
    *ch = F2H(gcg);
    *oh = F2H(gog);

    if (grad_cx_ptr != nullptr) {
      gcx = gcx * H2F(fg);
      T* gi = &(grad_cx_ptr[linearIndex]);
      *gi = F2H(gcx);
    }
  }
}

template<typename T>
struct FusedLstmCellFunctor final {
  void operator()(ep::Stream* stream, const int64_t cx_numel, const int64_t workspace_numel,
                  const int64_t hidden_size, const T* input_gates_ptr, const T* hidden_gates_ptr,
                  const T* cx_ptr, const T* input_bias_ptr, const T* hidden_bias_ptr, T* hy_ptr,
                  T* cy_ptr, T* workspace_ptr) {
    using ACC_T = acc_type<T>;
    if (workspace_numel < std::numeric_limits<int32_t>::max()) {
      RUN_CUDA_KERNEL((lstm_cell_forward<T, ACC_T, int32_t>), stream, cx_numel,
                      static_cast<int32_t>(cx_numel), static_cast<int32_t>(hidden_size),
                      input_gates_ptr, hidden_gates_ptr, cx_ptr, input_bias_ptr, hidden_bias_ptr,
                      hy_ptr, cy_ptr, workspace_ptr);
    } else {
      RUN_CUDA_KERNEL((lstm_cell_forward<T, ACC_T, int64_t>), stream, cx_numel, cx_numel,
                      hidden_size, input_gates_ptr, hidden_gates_ptr, cx_ptr, input_bias_ptr,
                      hidden_bias_ptr, hy_ptr, cy_ptr, workspace_ptr);
    }
  }
};

template<>
void FusedLstmCellFunctor<float16>::operator()(
    ep::Stream* stream, const int64_t cx_numel, const int64_t workspace_numel,
    const int64_t hidden_size, const float16* input_gates_ptr, const float16* hidden_gates_ptr,
    const float16* cx_ptr, const float16* input_bias_ptr, const float16* hidden_bias_ptr,
    float16* hy_ptr, float16* cy_ptr, float16* workspace_ptr) {
  if (workspace_numel < std::numeric_limits<int32_t>::max()) {
    RUN_CUDA_KERNEL(
        (lstm_cell_forward<half, float, int32_t>), stream, cx_numel, static_cast<int32_t>(cx_numel),
        static_cast<int32_t>(hidden_size), reinterpret_cast<const half*>(input_gates_ptr),
        reinterpret_cast<const half*>(hidden_gates_ptr), reinterpret_cast<const half*>(cx_ptr),
        reinterpret_cast<const half*>(input_bias_ptr),
        reinterpret_cast<const half*>(hidden_bias_ptr), reinterpret_cast<half*>(hy_ptr),
        reinterpret_cast<half*>(cy_ptr), reinterpret_cast<half*>(workspace_ptr));
  } else {
    RUN_CUDA_KERNEL((lstm_cell_forward<half, float, int64_t>), stream, cx_numel, cx_numel,
                    hidden_size, reinterpret_cast<const half*>(input_gates_ptr),
                    reinterpret_cast<const half*>(hidden_gates_ptr),
                    reinterpret_cast<const half*>(cx_ptr),
                    reinterpret_cast<const half*>(input_bias_ptr),
                    reinterpret_cast<const half*>(hidden_bias_ptr), reinterpret_cast<half*>(hy_ptr),
                    reinterpret_cast<half*>(cy_ptr), reinterpret_cast<half*>(workspace_ptr));
  }
}

template<typename T>
struct FusedLstmCellGradFunctor final {
  void operator()(ep::Stream* stream, const int64_t cx_numel, const int64_t workspace_numel,
                  const int64_t hidden_size, const T* grad_hy_ptr, const T* grad_cy_ptr,
                  const T* cx_ptr, const T* cy_ptr, const T* workspace_ptr, T* grad_gates_ptr,
                  T* grad_cx_ptr) {
    using ACC_T = acc_type<T>;
    if (workspace_numel < std::numeric_limits<int32_t>::max()) {
      RUN_CUDA_KERNEL((lstm_cell_backward<T, ACC_T, int32_t>), stream, cx_numel,
                      static_cast<int32_t>(cx_numel), static_cast<int32_t>(hidden_size),
                      grad_hy_ptr, grad_cy_ptr, cx_ptr, cy_ptr, workspace_ptr, grad_gates_ptr,
                      grad_cx_ptr);
    } else {
      RUN_CUDA_KERNEL((lstm_cell_backward<T, ACC_T, int64_t>), stream, cx_numel, cx_numel,
                      hidden_size, grad_hy_ptr, grad_cy_ptr, cx_ptr, cy_ptr, workspace_ptr,
                      grad_gates_ptr, grad_cx_ptr);
    }
  }
};

template<>
void FusedLstmCellGradFunctor<float16>::operator()(
    ep::Stream* stream, const int64_t cx_numel, const int64_t workspace_numel,
    const int64_t hidden_size, const float16* grad_hy_ptr, const float16* grad_cy_ptr,
    const float16* cx_ptr, const float16* cy_ptr, const float16* workspace_ptr,
    float16* grad_gates_ptr, float16* grad_cx_ptr) {
  if (workspace_numel < std::numeric_limits<int32_t>::max()) {
    RUN_CUDA_KERNEL((lstm_cell_backward<half, float, int32_t>), stream, cx_numel,
                    static_cast<int32_t>(cx_numel), static_cast<int32_t>(hidden_size),
                    reinterpret_cast<const half*>(grad_hy_ptr),
                    reinterpret_cast<const half*>(grad_cy_ptr),
                    reinterpret_cast<const half*>(cx_ptr), reinterpret_cast<const half*>(cy_ptr),
                    reinterpret_cast<const half*>(workspace_ptr),
                    reinterpret_cast<half*>(grad_gates_ptr), reinterpret_cast<half*>(grad_cx_ptr));
  } else {
    RUN_CUDA_KERNEL((lstm_cell_backward<half, float, int64_t>), stream, cx_numel, cx_numel,
                    hidden_size, reinterpret_cast<const half*>(grad_hy_ptr),
                    reinterpret_cast<const half*>(grad_cy_ptr),
                    reinterpret_cast<const half*>(cx_ptr), reinterpret_cast<const half*>(cy_ptr),
                    reinterpret_cast<const half*>(workspace_ptr),
                    reinterpret_cast<half*>(grad_gates_ptr), reinterpret_cast<half*>(grad_cx_ptr));
  }
}

}  // namespace

template<typename T>
class GpuFusedLstmCellKernel final : public user_op::OpKernel {
 public:
  GpuFusedLstmCellKernel() = default;
  ~GpuFusedLstmCellKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* input_gates = ctx->Tensor4ArgNameAndIndex("input_gates", 0);
    const user_op::Tensor* hidden_gates = ctx->Tensor4ArgNameAndIndex("hidden_gates", 0);
    const user_op::Tensor* cx = ctx->Tensor4ArgNameAndIndex("cx", 0);
    user_op::Tensor* hy = ctx->Tensor4ArgNameAndIndex("hy", 0);
    user_op::Tensor* cy = ctx->Tensor4ArgNameAndIndex("cy", 0);
    user_op::Tensor* workspace = ctx->Tensor4ArgNameAndIndex("workspace", 0);

    const T* input_bias_ptr = nullptr;
    const T* hidden_bias_ptr = nullptr;
    if (ctx->has_input("input_bias", 0)) {
      CHECK(ctx->has_input("hidden_bias", 0));
      input_bias_ptr = ctx->Tensor4ArgNameAndIndex("input_bias", 0)->dptr<T>();
      hidden_bias_ptr = ctx->Tensor4ArgNameAndIndex("hidden_bias", 0)->dptr<T>();
    }
    const T* input_gates_ptr = input_gates->dptr<T>();
    const T* hidden_gates_ptr = hidden_gates->dptr<T>();
    const T* cx_ptr = cx->dptr<T>();

    T* hy_ptr = hy->mut_dptr<T>();
    T* cy_ptr = cy->mut_dptr<T>();
    T* workspace_ptr = workspace->mut_dptr<T>();
    const int64_t cx_numel = cx->shape_view().elem_cnt();
    const int64_t workspace_numel = workspace->shape_view().elem_cnt();
    const int64_t hidden_size = cx->shape_view().At(cx->shape_view().NumAxes() - 1);
    FusedLstmCellFunctor<T>()(ctx->stream(), cx_numel, workspace_numel, hidden_size,
                              input_gates_ptr, hidden_gates_ptr, cx_ptr, input_bias_ptr,
                              hidden_bias_ptr, hy_ptr, cy_ptr, workspace_ptr);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_LSTM_CELL_KERNEL(dtype)                                                  \
  REGISTER_USER_KERNEL("fused_lstm_cell")                                                       \
      .SetCreateFn<GpuFusedLstmCellKernel<dtype>>()                                             \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                          \
                       && (user_op::HobDataType("cx", 0) == GetDataType<dtype>::value)          \
                       && (user_op::HobDataType("input_gates", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("hidden_gates", 0) == GetDataType<dtype>::value))

REGISTER_FUSED_LSTM_CELL_KERNEL(float);
REGISTER_FUSED_LSTM_CELL_KERNEL(float16);

class GpuFusedLstmCellGradFloatKernel final : public user_op::OpKernel {
 public:
  GpuFusedLstmCellGradFloatKernel() = default;
  ~GpuFusedLstmCellGradFloatKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* grad_hy = ctx->Tensor4ArgNameAndIndex("grad_hy", 0);
    const user_op::Tensor* grad_cy = ctx->Tensor4ArgNameAndIndex("grad_cy", 0);
    const user_op::Tensor* cx = ctx->Tensor4ArgNameAndIndex("cx", 0);
    const user_op::Tensor* cy = ctx->Tensor4ArgNameAndIndex("cy", 0);
    const user_op::Tensor* workspace = ctx->Tensor4ArgNameAndIndex("workspace", 0);
    user_op::Tensor* grad_gates = ctx->Tensor4ArgNameAndIndex("grad_gates", 0);
    user_op::Tensor* grad_cx = ctx->Tensor4ArgNameAndIndex("grad_cx", 0);

    const float* grad_hy_ptr = grad_hy->dptr<float>();
    const float* grad_cy_ptr = grad_cy->dptr<float>();
    const float* cx_ptr = cx->dptr<float>();
    const float* cy_ptr = cy->dptr<float>();
    const float* workspace_ptr = workspace->dptr<float>();

    float* grad_gates_ptr = grad_gates->mut_dptr<float>();
    float* grad_cx_ptr = nullptr;

    if (ctx->has_output("grad_cx", 0)) { grad_cx_ptr = grad_cx->mut_dptr<float>(); }

    const int64_t cx_numel = cx->shape_view().elem_cnt();
    const int64_t workspace_numel = workspace->shape_view().elem_cnt();
    const int64_t hidden_size = cx->shape_view().At(cx->shape_view().NumAxes() - 1);
    FusedLstmCellGradFunctor<float>()(ctx->stream(), cx_numel, workspace_numel, hidden_size,
                                      grad_hy_ptr, grad_cy_ptr, cx_ptr, cy_ptr, workspace_ptr,
                                      grad_gates_ptr, grad_cx_ptr);

    if (ctx->has_output("grad_bias", 0)) {
      float* grad_bias_ptr = ctx->Tensor4ArgNameAndIndex("grad_bias", 0)->mut_dptr<float>();
      std::vector<int32_t> axis;
      axis.push_back(0);
      const Shape& reduced_shape =
          CreateReducedShape(workspace->shape_view(), {axis.begin(), axis.end()});
      user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
      NdarrayReduce<DeviceType::kCUDA, float, BinaryFuncSum>::Reduce(
          ctx->stream(), XpuVarNdarray<float>(reduced_shape, grad_bias_ptr),
          XpuVarNdarray<const float>(grad_gates->shape_view(), grad_gates->dptr<float>()),
          XpuVarNdarray<float>(tmp_buffer->shape_view(), tmp_buffer->mut_dptr<float>()));
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("fused_lstm_cell_grad")
    .SetCreateFn<GpuFusedLstmCellGradFloatKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)
                     && (user_op::HobDataType("grad_hy", 0) == GetDataType<float>::value)
                     && (user_op::HobDataType("grad_cy", 0) == GetDataType<float>::value)
                     && (user_op::HobDataType("cx", 0) == GetDataType<float>::value)
                     && (user_op::HobDataType("cy", 0) == GetDataType<float>::value)
                     && (user_op::HobDataType("workspace", 0) == GetDataType<float>::value))
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) {
      size_t tmp_bytes = 0;
      if (ctx->has_output("grad_bias", 0)) {
        const Shape& in_shape = ctx->InputTensorDesc("workspace", 0).shape();
        tmp_bytes = GetCudaAlignedSize(in_shape.elem_cnt() * sizeof(float));
      } else {
        tmp_bytes = 0;
      }
      return tmp_bytes;
    });

class GpuFusedLstmCellGradHalfKernel final : public user_op::OpKernel {
 public:
  GpuFusedLstmCellGradHalfKernel() = default;
  ~GpuFusedLstmCellGradHalfKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* grad_hy = ctx->Tensor4ArgNameAndIndex("grad_hy", 0);
    const user_op::Tensor* grad_cy = ctx->Tensor4ArgNameAndIndex("grad_cy", 0);
    const user_op::Tensor* cx = ctx->Tensor4ArgNameAndIndex("cx", 0);
    const user_op::Tensor* cy = ctx->Tensor4ArgNameAndIndex("cy", 0);
    const user_op::Tensor* workspace = ctx->Tensor4ArgNameAndIndex("workspace", 0);
    user_op::Tensor* grad_gates = ctx->Tensor4ArgNameAndIndex("grad_gates", 0);
    user_op::Tensor* grad_cx = ctx->Tensor4ArgNameAndIndex("grad_cx", 0);

    const float16* grad_hy_ptr = grad_hy->dptr<float16>();
    const float16* grad_cy_ptr = grad_cy->dptr<float16>();
    const float16* cx_ptr = cx->dptr<float16>();
    const float16* cy_ptr = cy->dptr<float16>();
    const float16* workspace_ptr = workspace->dptr<float16>();

    float16* grad_gates_ptr = grad_gates->mut_dptr<float16>();
    float16* grad_cx_ptr = nullptr;

    if (ctx->has_output("grad_cx", 0)) { grad_cx_ptr = grad_cx->mut_dptr<float16>(); }

    const int64_t cx_numel = cx->shape_view().elem_cnt();
    const int64_t workspace_numel = workspace->shape_view().elem_cnt();
    const int64_t hidden_size = cx->shape_view().At(cx->shape_view().NumAxes() - 1);
    FusedLstmCellGradFunctor<float16>()(ctx->stream(), cx_numel, workspace_numel, hidden_size,
                                        grad_hy_ptr, grad_cy_ptr, cx_ptr, cy_ptr, workspace_ptr,
                                        grad_gates_ptr, grad_cx_ptr);

    if (ctx->has_output("grad_bias", 0)) {
      std::vector<int32_t> axis;
      axis.push_back(0);
      user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
      const ShapeView& in_shape = grad_gates->shape_view();
      const Shape& reduced_shape = CreateReducedShape(in_shape, {axis.begin(), axis.end()});
      float* in_tmp_buffer = tmp_buffer->mut_dptr<float>();
      const size_t in_tmp_buffer_bytes = GetCudaAlignedSize(in_shape.elem_cnt() * sizeof(float));
      float* out_tmp_buffer =
          reinterpret_cast<float*>(tmp_buffer->mut_dptr<char>() + in_tmp_buffer_bytes);
      const size_t out_tmp_buffer_bytes =
          GetCudaAlignedSize(reduced_shape.elem_cnt() * sizeof(float));
      float* reduce_tmp_buffer = reinterpret_cast<float*>(
          tmp_buffer->mut_dptr<char>() + in_tmp_buffer_bytes + out_tmp_buffer_bytes);
      const size_t reduce_tmp_buffer_bytes =
          GetCudaAlignedSize(in_shape.elem_cnt() * sizeof(float));
      CHECK_LE(in_tmp_buffer_bytes + out_tmp_buffer_bytes + reduce_tmp_buffer_bytes,
               tmp_buffer->shape_view().elem_cnt());
      auto h2f = ep::primitive::NewPrimitive<ep::primitive::CastFactory>(
          ctx->device_type(), DataType::kFloat16, DataType::kFloat);
      CHECK(h2f);
      auto f2h = ep::primitive::NewPrimitive<ep::primitive::CastFactory>(
          ctx->device_type(), DataType::kFloat, DataType::kFloat16);
      CHECK(f2h);
      h2f->Launch(ctx->stream(), grad_gates->dptr<float16>(), in_tmp_buffer, in_shape.elem_cnt());

      NdarrayReduce<DeviceType::kCUDA, float, BinaryFuncSum>::Reduce(
          ctx->stream(), XpuVarNdarray<float>(reduced_shape, out_tmp_buffer),
          XpuVarNdarray<const float>(in_shape, in_tmp_buffer),
          XpuVarNdarray<float>(in_shape, reduce_tmp_buffer));

      user_op::Tensor* output_tensor = ctx->Tensor4ArgNameAndIndex("grad_bias", 0);
      f2h->Launch(ctx->stream(), out_tmp_buffer, output_tensor->mut_dptr<float16>(),
                  output_tensor->shape_view().elem_cnt());
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("fused_lstm_cell_grad")
    .SetCreateFn<GpuFusedLstmCellGradHalfKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)
                     && (user_op::HobDataType("grad_hy", 0) == GetDataType<float16>::value)
                     && (user_op::HobDataType("grad_cy", 0) == GetDataType<float16>::value)
                     && (user_op::HobDataType("cx", 0) == GetDataType<float16>::value)
                     && (user_op::HobDataType("cy", 0) == GetDataType<float16>::value)
                     && (user_op::HobDataType("workspace", 0) == GetDataType<float16>::value))
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) {
      size_t tmp_bytes = 0;
      if (ctx->has_output("grad_bias", 0)) {
        const Shape& in_shape = ctx->InputTensorDesc("workspace", 0).shape();
        const Shape& out_shape = ctx->OutputTensorDesc("grad_bias", 0).shape();
        tmp_bytes = (2 * GetCudaAlignedSize(in_shape.elem_cnt() * sizeof(float))
                     + GetCudaAlignedSize(out_shape.elem_cnt() * sizeof(float)));
      } else {
        tmp_bytes = 0;
      }
      return tmp_bytes;
    });

}  // namespace oneflow
