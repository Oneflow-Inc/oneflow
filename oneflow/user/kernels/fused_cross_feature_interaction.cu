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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

enum InteractionMode { kVector = 0, kMatrix };

constexpr int kBlockSize = 256;

void InferMatmulMNK(const ShapeView& a_shape, const ShapeView& b_shape, bool transpose_a,
                    bool transpose_b, size_t* m, size_t* n, size_t* k) {
  const int64_t num_a_axes = a_shape.NumAxes();
  CHECK_GE(num_a_axes, 2);
  const int64_t num_b_axes = b_shape.NumAxes();
  CHECK_GE(num_b_axes, 2);
  if (!transpose_a) {
    *m = a_shape.At(num_a_axes - 2);
    *k = a_shape.At(num_a_axes - 1);
  } else {
    *m = a_shape.At(num_a_axes - 1);
    *k = a_shape.At(num_a_axes - 2);
  }
  if (!transpose_b) {
    CHECK_EQ(b_shape.At(num_b_axes - 2), *k);
    *n = b_shape.At(num_b_axes - 1);
  } else {
    CHECK_EQ(b_shape.At(num_b_axes - 1), *k);
    *n = b_shape.At(num_b_axes - 2);
  }
}

ep::primitive::BlasTransposeType GetBlasTransposeType(bool transpose) {
  return transpose ? ep::primitive::BlasTransposeType::T : ep::primitive::BlasTransposeType::N;
}

std::unique_ptr<ep::primitive::Matmul> NewMatmulPrimitive(DeviceType device_type,
                                                          DataType data_type, bool transpose_a,
                                                          bool transpose_b) {
  const auto trans_a = GetBlasTransposeType(transpose_a);
  const auto trans_b = GetBlasTransposeType(transpose_b);
  return ep::primitive::NewPrimitive<ep::primitive::MatmulFactory>(device_type, data_type, trans_a,
                                                                   trans_b);
}

template<typename Context>
std::unique_ptr<ep::primitive::Matmul> NewMatmulPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("x", 0)->data_type();
  return NewMatmulPrimitive(ctx->device_type(), data_type, /*transpose_a=*/false,
                            /*transpose_b=*/true);
}

auto MatmulPrimitiveExists() {
  return hob::make_custom("MatmulPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewMatmulPrimitive(&ctx).operator bool();
  });
}

template<typename T, typename IndexType, int pack_size, InteractionMode mode>
__global__ void FusedBiasAddMulAddResidualKernel(const T* in, const T* x, const T* x0,
                                                 const T* bias, T* out, const IndexType cols,
                                                 const IndexType elem_cnt) {
  const IndexType global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  using LoadPack = cuda::elementwise::Packed<T, pack_size>;
  for (IndexType linear_index = global_thread_id * pack_size,
                 step = gridDim.x * blockDim.x * pack_size;
       linear_index < elem_cnt; linear_index += step) {
    const IndexType row_idx = linear_index / cols;
    const IndexType col_idx = linear_index - row_idx * cols;

    const LoadPack* x0_load = reinterpret_cast<const LoadPack*>(x0 + linear_index);
    const LoadPack* x_load = reinterpret_cast<const LoadPack*>(x + linear_index);
    const LoadPack* bias_load = reinterpret_cast<const LoadPack*>(bias + col_idx);

    LoadPack x0_vec = *x0_load;
    LoadPack x_vec = *x_load;
    LoadPack bias_vec = *bias_load;

    LoadPack out_store;
    if (mode == InteractionMode::kVector) {
      T in_val = in[row_idx];
#pragma unroll
      for (int i = 0; i < pack_size; i++) {
        out_store.elem[i] = x0_vec.elem[i] * in_val + bias_vec.elem[i] + x_vec.elem[i];
      }
    } else if (mode == InteractionMode::kMatrix) {
      const LoadPack* in_load = reinterpret_cast<const LoadPack*>(in + linear_index);
      LoadPack in_vec = *in_load;
#pragma unroll
      for (int i = 0; i < pack_size; i++) {
        out_store.elem[i] = (in_vec.elem[i] + bias_vec.elem[i]) * x0_vec.elem[i] + x_vec.elem[i];
      }
    } else {
      __trap();
    }
    *(reinterpret_cast<LoadPack*>(out + linear_index)) = out_store;
  }
}

template<typename T>
int GetLaunchPackSize(const int64_t cols) {
  constexpr int type_pack_size = cuda::elementwise::PackSize<T>();
  for (int launch_pack_size = 8; launch_pack_size > 0; launch_pack_size /= 2) {
    if (type_pack_size >= launch_pack_size && cols % launch_pack_size == 0) {
      return launch_pack_size;
    }
  }
  return 1;
}

template<typename T, typename IndexType, InteractionMode mode>
void DispatchFusedBiasAddMulAddResidualPackSize(ep::Stream* stream, const T* in, const T* x,
                                                const T* x0, const T* bias, T* out,
                                                const IndexType cols, const IndexType elem_cnt) {
  int grid_size;
  const int pack_size = GetLaunchPackSize<T>(cols);
  const int64_t pack_num = elem_cnt / pack_size;
  cudaError_t err = cuda::elementwise::GetNumBlocks(pack_num, &grid_size);
  if (pack_size == 8) {
    FusedBiasAddMulAddResidualKernel<T, IndexType, 8, mode>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            in, x, x0, bias, out, cols, elem_cnt);
  } else if (pack_size == 4) {
    FusedBiasAddMulAddResidualKernel<T, IndexType, 4, mode>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            in, x, x0, bias, out, cols, elem_cnt);
  } else if (pack_size == 2) {
    FusedBiasAddMulAddResidualKernel<T, IndexType, 2, mode>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            in, x, x0, bias, out, cols, elem_cnt);
  } else {
    FusedBiasAddMulAddResidualKernel<T, IndexType, 1, mode>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
            in, x, x0, bias, out, cols, elem_cnt);
  }
}

template<typename T, InteractionMode mode>
void DispatchFusedBiasAddMulAddResidualIndexType(ep::Stream* stream, const T* in, const T* x,
                                                 const T* x0, const T* bias, T* out,
                                                 const int64_t cols, const int64_t elem_cnt) {
  if (elem_cnt < GetMaxVal<int32_t>()) {
    DispatchFusedBiasAddMulAddResidualPackSize<T, int32_t, mode>(stream, in, x, x0, bias, out, cols,
                                                                 elem_cnt);
  } else {
    DispatchFusedBiasAddMulAddResidualPackSize<T, int64_t, mode>(stream, in, x, x0, bias, out, cols,
                                                                 elem_cnt);
  }
}

template<typename T>
class FusedCrossFeatureInteractionKernel final : public user_op::OpKernel,
                                                 public user_op::CudaGraphSupport {
 public:
  FusedCrossFeatureInteractionKernel() = default;
  ~FusedCrossFeatureInteractionKernel() override = default;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    /*
    Cross Interaction v1:
    1. x matmul weight. matmul_result0 -> (B, E) matmul (1, E) -> (B, 1)
       dx = dmatmul_result0 matmul weight
       dw = x matmul dmatmul_result0

    2. matmul_result0 broadcast_mul x0. matmul_result1 -> (B, 1) broadcast_mul (B, E) -> (B, E)
       dmatmul_result0 = reduce_sum(dmatmul_result1 * x0, axis=1)
       dx0 = dmatmul_result1 broadcast_mul matmul_result0

    3. matmul_result1 broadcast_add bias. matmul_result2 -> (B, E) broadcast_add (1, E) -> (B, E)
       dmatmul_result1 = dout
       dbias = reduce_sum(dmatmul_result2, axis=0)

    4. matmul_result2 add x. out -> (B, E) elementwise_add (B, E) -> (B, E)
       dmatmul_result2 = dout, dx = dout.

    Cross Interaction Grad:
    dw = x matmul dmatmul_result0
    dx0 = dmatmul_result1 broadcast_mul matmul_result0
    dbias = reduce_sum(dmatmul_result2, axis=0)
    dx = (dmatmul_result0 matmul weight) + dout.

    Cross Interaction v2:
    1. x matmul weight. matmul_result0 -> (B, E) matmul (E, E) -> (B, E)

    2. matmul_result0 add bias. matmul_result1 -> (B, E) bias_add (1, E) -> (B, E)

    3. matmul_result1 multiply x0. matmul_result2 -> (B, E) elementwise_mul (B, E) -> (B, E)

    4. matmul_result2 add x. out -> (B, E) elementwise_add (B, E) -> (B, E)

    */
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const user_op::Tensor* x0 = ctx->Tensor4ArgNameAndIndex("x0", 0);
    const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* matmul_result = ctx->Tensor4ArgNameAndIndex("matmul_result", 0);
    const std::string interaction_mode = ctx->Attr<std::string>("interaction_mode");

    CHECK_EQ(out->shape_view().NumAxes(), 2);
    size_t m = 0, n = 0, k = 0;
    InferMatmulMNK(x->shape_view(), weight->shape_view(), /*trans_a=*/false, /*trans_b=*/true, &m,
                   &n, &k);
    const double alpha = 1.0;
    double beta = 0.0;
    auto matmul = NewMatmulPrimitive(ctx);
    CHECK(matmul);
    matmul->Launch(ctx->stream(), m, n, k, alpha, x->dptr(), weight->dptr(), beta,
                   matmul_result->mut_dptr());
    const int64_t elem_cnt = out->shape_view().elem_cnt();
    const int64_t cols = out->shape_view().At(1);
    if (interaction_mode == "vector") {
      DispatchFusedBiasAddMulAddResidualIndexType<T, InteractionMode::kVector>(
          ctx->stream(), matmul_result->mut_dptr<T>(), x->dptr<T>(), x0->dptr<T>(), bias->dptr<T>(),
          out->mut_dptr<T>(), cols, elem_cnt);
    } else {
      DispatchFusedBiasAddMulAddResidualIndexType<T, InteractionMode::kMatrix>(
          ctx->stream(), matmul_result->mut_dptr<T>(), x->dptr<T>(), x0->dptr<T>(), bias->dptr<T>(),
          out->mut_dptr<T>(), cols, elem_cnt);
    }
  }
};

#define REGISTER_FUSED_CROSS_FEATURE_INTERACTION_KERNEL(dtype)                        \
  REGISTER_USER_KERNEL("fused_cross_feature_interaction")                             \
      .SetCreateFn<FusedCrossFeatureInteractionKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value) \
                       && MatmulPrimitiveExists());

REGISTER_FUSED_CROSS_FEATURE_INTERACTION_KERNEL(float)
REGISTER_FUSED_CROSS_FEATURE_INTERACTION_KERNEL(half)

}  // namespace

}  // namespace oneflow
