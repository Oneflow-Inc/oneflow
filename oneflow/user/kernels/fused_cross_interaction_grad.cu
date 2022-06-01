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
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/cuda/elementwise.cuh"

namespace oneflow {

namespace {

constexpr int kBlockSize = 256;

void InferMatmulMNK(const DimVector& a_shape, const DimVector& b_shape, bool transpose_a,
                    bool transpose_b, size_t* m, size_t* n, size_t* k) {
  const int64_t num_a_axes = a_shape.size();
  CHECK_GE(num_a_axes, 2);
  const int64_t num_b_axes = b_shape.size();
  CHECK_GE(num_b_axes, 2);
  if (!transpose_a) {
    *m = a_shape.at(num_a_axes - 2);
    *k = a_shape.at(num_a_axes - 1);
  } else {
    *m = a_shape.at(num_a_axes - 1);
    *k = a_shape.at(num_a_axes - 2);
  }
  if (!transpose_b) {
    CHECK_EQ(b_shape.at(num_b_axes - 2), *k);
    *n = b_shape.at(num_b_axes - 1);
  } else {
    CHECK_EQ(b_shape.at(num_b_axes - 1), *k);
    *n = b_shape.at(num_b_axes - 2);
  }
}

ep::primitive::BlasTransposeType GetBlasTransposeType(bool transpose) {
  return transpose ? ep::primitive::BlasTransposeType::T : ep::primitive::BlasTransposeType::N;
}

template<typename T>
struct MulOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a * b; }
};

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

template<typename T, typename IndexType, int pack_size>
__global__ void BroadcastMulAddResidualKernel(const T* x, const T* y, T* out, const IndexType cols,
                                              const IndexType elem_cnt) {
  const IndexType global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  using LoadType = cuda::elementwise::PackType<T, pack_size>;
  using LoadPack = cuda::elementwise::Pack<T, pack_size>;
  for (IndexType linear_index = global_thread_id * pack_size,
                 step = gridDim.x * blockDim.x * pack_size;
       linear_index < elem_cnt; linear_index += step) {
    const IndexType row_idx = linear_index / cols;
    const LoadType* x_load = reinterpret_cast<const LoadType*>(x + linear_index);
    LoadPack x_vec;
    x_vec.storage = *x_load;

    LoadPack out_vec;
    const T y_val = y[row_idx];
#pragma unroll
    for (int i = 0; i < pack_size; i++) { out_vec.elem[i] = x_vec.elem[i] * y_val + x_vec.elem[i]; }
    *(reinterpret_cast<LoadType*>(out + linear_index)) = out_vec.storage;
  }
}

template<typename T, typename IndexType>
void DispatchBroadcastMulAddResidualPackSize(ep::Stream* stream, const T* x, const T* y, T* out,
                                             const IndexType cols, const IndexType elem_cnt) {
  int grid_size;
  const int pack_size = GetLaunchPackSize<T>(cols);
  const int64_t pack_num = elem_cnt / pack_size;
  cudaError_t err = cuda::elementwise::GetNumBlocks(pack_num, &grid_size);
  if (pack_size == 8) {
    BroadcastMulAddResidualKernel<T, IndexType, 8>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(x, y, out, cols,
                                                                                    elem_cnt);
  } else if (pack_size == 4) {
    BroadcastMulAddResidualKernel<T, IndexType, 4>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(x, y, out, cols,
                                                                                    elem_cnt);
  } else if (pack_size == 2) {
    BroadcastMulAddResidualKernel<T, IndexType, 2>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(x, y, out, cols,
                                                                                    elem_cnt);
  } else {
    BroadcastMulAddResidualKernel<T, IndexType, 1>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(x, y, out, cols,
                                                                                    elem_cnt);
  }
}

template<typename T>
void DispatchBroadcastMulAddResidualIndexType(ep::Stream* stream, const T* x, const T* y, T* out,
                                              const int64_t cols, const int64_t elem_cnt) {
  if (elem_cnt < GetMaxVal<int32_t>()) {
    DispatchBroadcastMulAddResidualPackSize<T, int32_t>(stream, x, y, out, cols, elem_cnt);
  } else {
    DispatchBroadcastMulAddResidualPackSize<T, int64_t>(stream, x, y, out, cols, elem_cnt);
  }
}

}  // namespace

namespace user_op {

std::unique_ptr<ep::primitive::Matmul> NewMatmulPrimitive(DeviceType device_type,
                                                          DataType data_type, bool transpose_a,
                                                          bool transpose_b) {
  const auto trans_a = GetBlasTransposeType(transpose_a);
  const auto trans_b = GetBlasTransposeType(transpose_b);
  return ep::primitive::NewPrimitive<ep::primitive::MatmulFactory>(device_type, data_type, trans_a,
                                                                   trans_b);
}

template<typename Context>
std::unique_ptr<ep::primitive::Matmul> NewReduceMatmulPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("dy", 0)->data_type();
  return NewMatmulPrimitive(ctx->device_type(), data_type, /*transpose_a=*/false,
                            /*transpose_b=*/false);
}

auto ReduceMatmulPrimitiveExists() {
  return hob::make_custom("MatmulPrimitiveExists", [](const KernelRegContext& ctx) {
    return NewReduceMatmulPrimitive(&ctx).operator bool();
  });
}

template<typename Context>
std::unique_ptr<ep::primitive::Matmul> NewWeightGradMatmulPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("x", 0)->data_type();
  return NewMatmulPrimitive(ctx->device_type(), data_type, /*transpose_a=*/true,
                            /*transpose_b=*/false);
}

auto WeightGradMatmulPrimitiveExists() {
  return hob::make_custom("MatmulPrimitiveExists", [](const KernelRegContext& ctx) {
    return NewWeightGradMatmulPrimitive(&ctx).operator bool();
  });
}

template<typename T>
class FusedCrossInteractionGradKernel final : public OpKernel, public CudaGraphSupport {
 public:
  FusedCrossInteractionGradKernel() = default;
  ~FusedCrossInteractionGradKernel() = default;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  using user_op::OpKernel::Compute;
  void Compute(KernelComputeContext* ctx) const override {
    /*
    Cross Interaction:
    1. x matmul weight. matmul_result -> (B, E) matmul (1, E) -> (B, 1)
    2. matmul_result broadcast_mul x_0. matmul_result1 -> (B, 1) broadcast_mul (B, E) -> (B, E)
    3. matmul_result1 broadcast_add bias. matmul_result2 -> (B, E) broadcast_add (1, E) -> (B, E)
    4. matmul_result2 add x. out -> (B, E) elementwise_add (B, E) -> (B, E)

    Cross Interaction Grad:
    1. d_bias = reduce_sum(dy, axis=0). (B, E) reduce_sum(axis=0) -> (1, E)
    2. d_matmul_result = reduce_sum(dy, axis=0). (B, E) reduce_sum(axis=0) -> (1, E)
    1. d_bias = reduce_sum(dy, axis=0). (B, E) reduce_sum(axis=0) -> (1, E)
    1. d_bias = reduce_sum(dy, axis=0). (B, E) reduce_sum(axis=0) -> (1, E)
    1. d_bias = reduce_sum(dy, axis=0). (B, E) reduce_sum(axis=0) -> (1, E)


    */

    const Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const Tensor* x_0 = ctx->Tensor4ArgNameAndIndex("x_0", 0);
    const Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const Tensor* matmul_result = ctx->Tensor4ArgNameAndIndex("matmul_result", 0);

    const int64_t batch_size = dy->shape().At(0);
    const int64_t hidden_size = dy->shape().At(1);
    const int64_t out_size = weight->shape().At(0);
    const int64_t dy_elem_cnt = dy->shape().elem_cnt();

    Tensor* dx_0 = ctx->Tensor4ArgNameAndIndex("dx_0", 0);
    Tensor* dw = ctx->Tensor4ArgNameAndIndex("dw", 0);
    Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    Tensor* dbias = ctx->Tensor4ArgNameAndIndex("dbias", 0);
    Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    // step1: Get Dbias.
    const T* ones = nullptr;
    auto* cuda_device = dynamic_cast<ep::CudaDevice*>(ctx->stream()->device());
    if (cuda_device != nullptr) {
      ones = static_cast<const T*>(cuda_device->GetConstOnes(dy->data_type(), batch_size));
    }
    size_t m = 0, n = 0, k = 0;
    DimVector dy_shape(2);
    dy->shape().ToDimVector(&dy_shape);
    DimVector ones_buf_shape(2);
    ones_buf_shape.at(0) = 1;
    ones_buf_shape.at(1) = batch_size;
    InferMatmulMNK(ones_buf_shape, dy_shape, /*trans_a=*/false, /*trans_b=*/false, &m, &n, &k);
    auto reduce_matmul = NewReduceMatmulPrimitive(ctx);
    CHECK(reduce_matmul);
    reduce_matmul->Launch(ctx->stream(), m, n, k, 1.0, ones, dy->dptr(), 0.0, dbias->mut_dptr());

    // step2: Get Dt.
    T* dy_mul_x0 = reinterpret_cast<T*>(tmp_buffer->mut_dptr());
    T* dt = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>()
                                 + GetCudaAlignedSize(dy_elem_cnt * sizeof(T)));
    OF_CUDA_CHECK(cuda::elementwise::Binary(MulOp<T>(), dy_elem_cnt, dy_mul_x0, dy->dptr<T>(),
                                            x_0->dptr<T>(),
                                            ctx->stream()->As<ep::CudaStream>()->cuda_stream()));

    ones = static_cast<const T*>(cuda_device->GetConstOnes(dy->data_type(), hidden_size));
    DimVector dy_mul_x0_shape(2);
    dy->shape().ToDimVector(&dy_mul_x0_shape);
    ones_buf_shape.at(0) = hidden_size;
    ones_buf_shape.at(1) = 1;
    InferMatmulMNK(dy_mul_x0_shape, ones_buf_shape, /*trans_a=*/false, /*trans_b=*/false, &m, &n,
                   &k);
    reduce_matmul->Launch(ctx->stream(), m, n, k, 1.0, dy_mul_x0, ones, 0.0, dt);

    // step3: Get dxi
    DimVector dt_shape(2);
    dt_shape.at(0) = batch_size;
    dt_shape.at(1) = 1;
    DimVector weight_shape(2);
    weight->shape().ToDimVector(&weight_shape);

    InferMatmulMNK(dt_shape, weight_shape, /*trans_a=*/false, /*trans_b=*/false, &m, &n, &k);
    reduce_matmul->Launch(ctx->stream(), m, n, k, 1.0, dt, weight->dptr(), 0.0, dx->mut_dptr<T>());

    // step4: Get Dw.
    DimVector x_shape(2);
    x->shape().ToDimVector(&x_shape);

    InferMatmulMNK(x_shape, dt_shape, /*trans_a=*/true, /*trans_b=*/false, &m, &n, &k);
    auto weight_grad_matmul = NewWeightGradMatmulPrimitive(ctx);
    CHECK(weight_grad_matmul);
    weight_grad_matmul->Launch(ctx->stream(), m, n, k, 1.0, x->dptr(), dt, 0.0, dw->mut_dptr());

    // step5: Get Dx0.
    DispatchBroadcastMulAddResidualIndexType<T>(ctx->stream(), dy->dptr<T>(),
                                                matmul_result->dptr<T>(), dx_0->mut_dptr<T>(),
                                                hidden_size, dy->shape().elem_cnt());
  }
};

#define REGISTER_FUSED_CROSS_INTERACTION_GRAD_KERNEL(dtype)                                   \
  REGISTER_USER_KERNEL("fused_cross_interaction_grad")                                        \
      .SetCreateFn<FusedCrossInteractionGradKernel<dtype>>()                                  \
      .SetIsMatchedHob((HobDeviceType() == DeviceType::kCUDA)                                 \
                       && (HobDataType("dy", 0) == GetDataType<dtype>::value)                 \
                       && ReduceMatmulPrimitiveExists() && WeightGradMatmulPrimitiveExists()) \
      .SetInferTmpSizeFn([](InferContext* ctx) {                                              \
        size_t tmp_size = 0;                                                                  \
        const TensorDesc& dy = ctx->InputTensorDesc("dy", 0);                                 \
        const int64_t dy_elem_cnt = dy.shape().elem_cnt();                                    \
        const int64_t batch_size = dy.shape().At(0);                                          \
        size_t dy_mul_x0_size = GetCudaAlignedSize(dy_elem_cnt * sizeof(dtype));              \
        size_t dt_size = GetCudaAlignedSize(batch_size * sizeof(dtype));                      \
        tmp_size = dy_mul_x0_size + dt_size;                                                  \
        return tmp_size;                                                                      \
      });

REGISTER_FUSED_CROSS_INTERACTION_GRAD_KERNEL(float)
REGISTER_FUSED_CROSS_INTERACTION_GRAD_KERNEL(half)

}  // namespace user_op

}  // namespace oneflow
