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
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/ep/include/primitive/binary_op.h"
#include "oneflow/core/ep/common/primitive/binary_functor.h"
#include "oneflow/core/ep/cuda/primitive/binary_functor.cuh"
#include "oneflow/core/ep/include/primitive/unary_op.h"
#include "oneflow/core/ep/common/primitive/unary_functor.h"
#include "oneflow/core/ep/cuda/primitive/unary_functor.cuh"

#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000
#include "oneflow/core/device/cuda_pseudo_bfloat16.h"

namespace oneflow {

namespace {

// declear using "BinaryFunctor" from namespace "ep::primitive::broadcast_elementwise_binary"
template<DeviceType device, ep::primitive::BinaryOp binary_op, typename Src, typename Dst>
using BinaryFunctor =
    ep::primitive::broadcast_elementwise_binary::BinaryFunctor<device, binary_op, Src, Dst>;

template<typename T, typename IndexType, ep::primitive::BinaryOp d_act_type,
         ep::primitive::UnaryOp act_type, int32_t pack_size>
__global__ void FusedGluWithoutLinearGradGpu(
    const IndexType m, const IndexType packed_n, const IndexType pack_num,
    const IndexType packed_stride, BinaryFunctor<DeviceType::kCUDA, d_act_type, T, T> dact,
    ep::primitive::UnaryFunctor<DeviceType::kCUDA, act_type, T, T> act, const T* dy,
    const T* matmul_wx, const T* matmul_vx, T* d_matmul_wx, T* d_matmul_vx) {
  // define type of Pack
  using LoadPack = cuda::elementwise::Packed<T, pack_size>;

  // obtain global thread index
  IndexType global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  // workload of current thread
  for (IndexType packed_index = global_thread_id, step = gridDim.x * blockDim.x;
       packed_index < pack_num; packed_index += step) {
    // obtain the row and col index in output tensor "d_matmul_wx" and "d_matmul_vx"
    const IndexType packed_row = packed_index / packed_n;
    const IndexType packed_col = packed_index - packed_row * packed_n;

    // cast type to load type
    const LoadPack* dy_load =
        reinterpret_cast<const LoadPack*>(dy) + (packed_row * packed_n + packed_col);
    const LoadPack* matmul_wx_load =
        reinterpret_cast<const LoadPack*>(matmul_wx) + (packed_row * packed_stride + packed_col);
    const LoadPack* matmul_vx_load =
        reinterpret_cast<const LoadPack*>(matmul_vx) + (packed_row * packed_stride + packed_col);

    // init vectors
    LoadPack dy_vec = *dy_load;
    LoadPack matmul_wx_vec = *matmul_wx_load;
    LoadPack matmul_vx_vec = *matmul_vx_load;
    LoadPack d_matmul_wx_vec;
    LoadPack d_matmul_vx_vec;
#pragma unroll
    for (int i = 0; i < pack_size; i++) {
      // calculate the gradient of activated gate
      T d_act_gate = matmul_wx_vec.elem[i] * dy_vec.elem[i];

      // calculate the gradient of hidden_state
      T gate = matmul_vx_vec.elem[i];
      T act_gate = act(gate);
      d_matmul_wx_vec.elem[i] = act_gate * dy_vec.elem[i];  // d_hidden_state

      // calculate the gradient of gate
      d_matmul_vx_vec.elem[i] = dact(d_act_gate, gate);  // d_gate
    }
    *(reinterpret_cast<LoadPack*>(d_matmul_wx) + (packed_row * packed_stride + packed_col)) =
        d_matmul_wx_vec;
    *(reinterpret_cast<LoadPack*>(d_matmul_vx) + (packed_row * packed_stride + packed_col)) =
        d_matmul_vx_vec;
  }
}

template<typename T, typename IndexType, ep::primitive::UnaryOp act_type,
         ep::primitive::BinaryOp d_act_type, int32_t pack_size>
void LaunchFusedGluWithoutLinearGradGpu(ep::Stream* stream, const IndexType m,
                                        const IndexType packed_n, const IndexType pack_num,
                                        const IndexType packed_stride, const T* dy,
                                        const T* matmul_wx, const T* matmul_vx, T* d_matmul_wx,
                                        T* d_matmul_vx) {
  constexpr int32_t block_size = 128;
  unsigned int grid_size = (pack_num + block_size - 1) / block_size;
  ep::primitive::UnaryFunctor<DeviceType::kCUDA, act_type, T, T> act(0, 0);
  BinaryFunctor<DeviceType::kCUDA, d_act_type, T, T> dact(0, 0);
  FusedGluWithoutLinearGradGpu<T, IndexType, d_act_type, act_type, pack_size>
      <<<grid_size, block_size, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
          m, packed_n, pack_num, packed_stride, dact, act, dy, matmul_wx, matmul_vx, d_matmul_wx,
          d_matmul_vx);
}

template<typename T, ep::primitive::UnaryOp act_type, ep::primitive::BinaryOp d_act_type,
         int32_t pack_size>
void DispatchIndexType(ep::Stream* stream, const int64_t m, const int64_t packed_n,
                       const int64_t pack_num, const int64_t packed_stride, const T* dy,
                       const T* matmul_wx, const T* matmul_vx, T* d_matmul_wx, T* d_matmul_vx) {
  if (pack_num < (1 << 30)) {
    LaunchFusedGluWithoutLinearGradGpu<T, int32_t, act_type, d_act_type, pack_size>(
        stream, m, packed_n, pack_num, packed_stride, dy, matmul_wx, matmul_vx, d_matmul_wx,
        d_matmul_vx);
  } else {
    LaunchFusedGluWithoutLinearGradGpu<T, int64_t, act_type, d_act_type, pack_size>(
        stream, m, packed_n, pack_num, packed_stride, dy, matmul_wx, matmul_vx, d_matmul_wx,
        d_matmul_vx);
  }
}

template<typename T, ep::primitive::UnaryOp act_type, ep::primitive::BinaryOp d_act_type,
         int32_t alignment, typename std::enable_if<alignment / sizeof(T) == 0, int>::type = 0>
void DispatchPackSize(ep::Stream* stream, const int64_t m, const int64_t n, const int64_t stride,
                      const T* dy, const T* matmul_wx, const T* matmul_vx, T* d_matmul_wx,
                      T* d_matmul_vx) {
  DispatchIndexType<T, act_type, d_act_type, 1>(stream, m, n, m * n, stride, dy, matmul_wx,
                                                matmul_vx, d_matmul_wx, d_matmul_vx);
}

template<typename T, ep::primitive::UnaryOp act_type, ep::primitive::BinaryOp d_act_type,
         int32_t alignment, typename std::enable_if<alignment / sizeof(T) != 0, int>::type = 0>
void DispatchPackSize(ep::Stream* stream, const int64_t m, const int64_t n, const int64_t stride,
                      const T* dy, const T* matmul_wx, const T* matmul_vx, T* d_matmul_wx,
                      T* d_matmul_vx) {
  const int64_t pack_size = alignment / sizeof(T);
  const int64_t packed_n = n / pack_size;
  const int64_t pack_num = m * packed_n;
  const int64_t packed_stride = stride / pack_size;
  DispatchIndexType<T, act_type, d_act_type, alignment / sizeof(T)>(
      stream, m, packed_n, pack_num, packed_stride, dy, matmul_wx, matmul_vx, d_matmul_wx,
      d_matmul_vx);
}

template<typename T, ep::primitive::UnaryOp act_type, ep::primitive::BinaryOp d_act_type>
void DispatchAlignment(ep::Stream* stream, const int64_t m, const int64_t n, const int64_t stride,
                       const T* dy, const T* matmul_wx, const T* matmul_vx, T* d_matmul_wx,
                       T* d_matmul_vx) {
  const auto IsAligned = [&](const size_t alignment) {
    const uintptr_t dy_ptr = reinterpret_cast<uintptr_t>(dy);
    const uintptr_t matmul_wx_ptr = reinterpret_cast<uintptr_t>(matmul_wx);
    const uintptr_t matmul_vx_ptr = reinterpret_cast<uintptr_t>(matmul_vx);
    const uintptr_t d_matmul_wx_ptr = reinterpret_cast<uintptr_t>(d_matmul_wx);
    const uintptr_t d_matmul_vx_ptr = reinterpret_cast<uintptr_t>(d_matmul_vx);
    const int64_t pack_size = alignment / sizeof(T);
    return pack_size != 0 ? (/* memory address alignment */
                             dy_ptr % alignment == 0 && matmul_vx_ptr % alignment == 0
                             && matmul_wx_ptr % alignment == 0 && d_matmul_wx_ptr % alignment == 0
                             && d_matmul_vx_ptr % alignment == 0
                             /* #element per row alignment */
                             && n % (pack_size) == 0)
                          : false;
  };

  // dispatch alignment
  if (IsAligned(16)) {
    DispatchPackSize<T, act_type, d_act_type, 16>(stream, m, n, stride, dy, matmul_wx, matmul_vx,
                                                  d_matmul_wx, d_matmul_vx);
  } else if (IsAligned(8)) {
    DispatchPackSize<T, act_type, d_act_type, 8>(stream, m, n, stride, dy, matmul_wx, matmul_vx,
                                                 d_matmul_wx, d_matmul_vx);
  } else if (IsAligned(4)) {
    DispatchPackSize<T, act_type, d_act_type, 4>(stream, m, n, stride, dy, matmul_wx, matmul_vx,
                                                 d_matmul_wx, d_matmul_vx);
  } else if (IsAligned(2)) {
    DispatchPackSize<T, act_type, d_act_type, 2>(stream, m, n, stride, dy, matmul_wx, matmul_vx,
                                                 d_matmul_wx, d_matmul_vx);
  } else {
    DispatchPackSize<T, act_type, d_act_type, 1>(stream, m, n, stride, dy, matmul_wx, matmul_vx,
                                                 d_matmul_wx, d_matmul_vx);
  }
}

template<typename T>
void DispatchActivationType(ep::Stream* stream, const int64_t m, const int64_t n,
                            const std::string& activation, const int64_t stride, const T* dy,
                            const T* matmul_wx, const T* matmul_vx, T* d_matmul_wx,
                            T* d_matmul_vx) {
  if (activation == "none") {
    DispatchAlignment<T, ep::primitive::UnaryOp::kIdentity,
                      ep::primitive::BinaryOp::kIdentityBackwardWithDyX>(
        stream, m, n, stride, dy, matmul_wx, matmul_vx, d_matmul_wx, d_matmul_vx);
  } else if (activation == "sigmoid") {
    DispatchAlignment<T, ep::primitive::UnaryOp::kSigmoid,
                      ep::primitive::BinaryOp::kSigmoidBackwardWithDyX>(
        stream, m, n, stride, dy, matmul_wx, matmul_vx, d_matmul_wx, d_matmul_vx);
  } else if (activation == "relu") {
    DispatchAlignment<T, ep::primitive::UnaryOp::kRelu,
                      ep::primitive::BinaryOp::kReluBackwardWithDyX>(
        stream, m, n, stride, dy, matmul_wx, matmul_vx, d_matmul_wx, d_matmul_vx);
  } else if (activation == "gelu") {
    DispatchAlignment<T, ep::primitive::UnaryOp::kGelu,
                      ep::primitive::BinaryOp::kGeluBackwardWithDyX>(
        stream, m, n, stride, dy, matmul_wx, matmul_vx, d_matmul_wx, d_matmul_vx);
  } else if (activation == "fast_gelu") {
    DispatchAlignment<T, ep::primitive::UnaryOp::kFastGelu,
                      ep::primitive::BinaryOp::kFastGeluBackwardWithDyX>(
        stream, m, n, stride, dy, matmul_wx, matmul_vx, d_matmul_wx, d_matmul_vx);
  } else if (activation == "silu") {
    DispatchAlignment<T, ep::primitive::UnaryOp::kSilu,
                      ep::primitive::BinaryOp::kSiluBackwardWithDyX>(
        stream, m, n, stride, dy, matmul_wx, matmul_vx, d_matmul_wx, d_matmul_vx);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
class GpuFusedGluWithoutLinearGradKernel final : public user_op::OpKernel {
 public:
  GpuFusedGluWithoutLinearGradKernel() = default;
  ~GpuFusedGluWithoutLinearGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // obtain tensors from context
    const user_op::Tensor* input_tensor_dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* input_tensor_matmul_wx = ctx->Tensor4ArgNameAndIndex("matmul_wx", 0);
    user_op::Tensor* out_tensor_d_matmul_wx = ctx->Tensor4ArgNameAndIndex("d_matmul_wx", 0);

    // obtain optional tensors from context
    bool is_split_mode = false;
    user_op::Tensor* input_tensor_matmul_vx = nullptr;
    user_op::Tensor* out_tensor_d_matmul_vx = nullptr;
    if (ctx->has_input("matmul_vx", 0)) {
      input_tensor_matmul_vx = ctx->Tensor4ArgNameAndIndex("matmul_vx", 0);
      out_tensor_d_matmul_vx = ctx->Tensor4ArgNameAndIndex("d_matmul_vx", 0);
      is_split_mode = true;
    }

    // obtain tensor shapes and number of axes
    const ShapeView& dy_shape = input_tensor_dy->shape_view();
    const ShapeView& matmul_wx_shape = input_tensor_matmul_wx->shape_view();
    const ShapeView& d_matmul_wx_shape = out_tensor_d_matmul_wx->shape_view();
    const size_t dy_num_axes = dy_shape.NumAxes();
    const size_t matmul_wx_num_axes = matmul_wx_shape.NumAxes();

    // validate dimension and number of axes
    CHECK_GE(dy_num_axes, 2) << "number of axes of \'dy\' should have be greater than 1, yet get "
                             << dy_num_axes;
    CHECK_GE(matmul_wx_num_axes, 2)
        << "number of axes of \'matmul_wx\' should have be greater than 1, yet get "
        << matmul_wx_num_axes;
    CHECK_EQ(dy_num_axes, matmul_wx_num_axes)
        << "number of axes of \'dy\'(" << dy_num_axes
        << ") is not consistant with the one of \'matmul_wx\'(" << matmul_wx_num_axes << ")";

    // check input shape
    if (is_split_mode) {
      CHECK_EQ(dy_shape.At(dy_num_axes - 1), matmul_wx_shape.At(matmul_wx_num_axes - 1))
          << "the last dimension of \'dy\'(" << dy_shape.At(dy_num_axes - 1)
          << ") is not consistant with the last dimension of \'matmul_wx\'("
          << matmul_wx_shape.At(matmul_wx_num_axes - 1) << ")";
    } else {
      CHECK_EQ(2 * dy_shape.At(dy_num_axes - 1), matmul_wx_shape.At(matmul_wx_num_axes - 1))
          << "two times of the last dimension of \'dy\'(" << 2 * dy_shape.At(dy_num_axes - 1)
          << ") is not consistant with the last dimension of \'matmul_wx\'("
          << matmul_wx_shape.At(matmul_wx_num_axes - 1) << ")";
    }

    // check optional input tensor shapes
    if (is_split_mode) {
      const user_op::Tensor* input_tensor_matmul_vx = ctx->Tensor4ArgNameAndIndex("matmul_vx", 0);
      const ShapeView& matmul_vx_shape = input_tensor_matmul_vx->shape_view();
      const size_t matmul_vx_num_axes = matmul_vx_shape.NumAxes();
      CHECK_GE(matmul_vx_num_axes, 2)
          << "number of axes of \'matmul_vx\' should have be greater than 1, yet get "
          << matmul_vx_num_axes;
      CHECK_EQ(matmul_vx_num_axes, dy_num_axes)
          << "number of axes of \'dy\'(" << dy_num_axes
          << ") is not consistant with the one of \'matmul_vx\'(" << matmul_vx_num_axes << ")";
      CHECK_EQ(matmul_vx_shape.At(matmul_vx_num_axes - 1), dy_shape.At(dy_num_axes - 1))
          << "the last dimension of \'dy\'(" << dy_shape.At(dy_num_axes - 1)
          << ") is not consistant with the last dimension of \'matmul_vx\'("
          << matmul_vx_shape.At(matmul_vx_num_axes - 1) << ")";
    }

    // infer m, n
    const int64_t m = dy_shape.Count(0, dy_num_axes - 1);
    const int64_t n = dy_shape.At(dy_num_axes - 1);

    // start dispatch process
    DispatchActivationType<T>(
        ctx->stream(),
        /*m, n=*/m, n,
        /*activation=*/ctx->Attr<std::string>("activation"),
        /*stride=*/is_split_mode ? n : n * 2,
        /*dy=*/input_tensor_dy->dptr<T>(),
        /*matmul_wx=*/input_tensor_matmul_wx->dptr<T>(),
        /*matmul_vx=*/
        is_split_mode ? input_tensor_matmul_vx->dptr<T>() : input_tensor_matmul_wx->dptr<T>() + n,
        /*d_matmul_wx=*/out_tensor_d_matmul_wx->mut_dptr<T>(),
        /*d_matmul_vx=*/
        is_split_mode ? out_tensor_d_matmul_vx->mut_dptr<T>()
                      : out_tensor_d_matmul_wx->mut_dptr<T>() + n);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_GPU_FUSED_GLU_WITHOUT_LINEAR_GRAD_KERNEL(dtype)       \
  REGISTER_USER_KERNEL("fused_glu_without_linear_grad")                \
      .SetCreateFn<GpuFusedGluWithoutLinearGradKernel<dtype>>()        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("d_matmul_wx", 0) == GetDataType<dtype>::value));

REGISTER_GPU_FUSED_GLU_WITHOUT_LINEAR_GRAD_KERNEL(double)
REGISTER_GPU_FUSED_GLU_WITHOUT_LINEAR_GRAD_KERNEL(float)
REGISTER_GPU_FUSED_GLU_WITHOUT_LINEAR_GRAD_KERNEL(half)
#if CUDA_VERSION >= 11000
REGISTER_GPU_FUSED_GLU_WITHOUT_LINEAR_GRAD_KERNEL(nv_bfloat16)
#endif

}  // namespace oneflow
