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
struct AddOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
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
__global__ void BroadcastMulKernel(const T* x, const T* y, T* out, const IndexType cols,
                                   const IndexType elem_cnt) {
  const IndexType global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  using LoadPack = cuda::elementwise::Packed<T, pack_size>;
  for (IndexType linear_index = global_thread_id * pack_size,
                 step = gridDim.x * blockDim.x * pack_size;
       linear_index < elem_cnt; linear_index += step) {
    const IndexType row_idx = linear_index / cols;
    const LoadPack* x_load = reinterpret_cast<const LoadPack*>(x + linear_index);
    LoadPack x_vec = *x_load;
    LoadPack out_store;
    const T y_val = y[row_idx];
#pragma unroll
    for (int i = 0; i < pack_size; i++) { out_store.elem[i] = x_vec.elem[i] * y_val; }
    *(reinterpret_cast<LoadPack*>(out + linear_index)) = out_store;
  }
}

template<typename T, typename IndexType>
void DispatchBroadcastMulPackSize(ep::Stream* stream, const T* x, const T* y, T* out,
                                  const IndexType cols, const IndexType elem_cnt) {
  int grid_size;
  const int pack_size = GetLaunchPackSize<T>(cols);
  const int64_t pack_num = elem_cnt / pack_size;
  cudaError_t err = cuda::elementwise::GetNumBlocks(pack_num, &grid_size);
  if (pack_size == 8) {
    BroadcastMulKernel<T, IndexType, 8>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(x, y, out, cols,
                                                                                    elem_cnt);
  } else if (pack_size == 4) {
    BroadcastMulKernel<T, IndexType, 4>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(x, y, out, cols,
                                                                                    elem_cnt);
  } else if (pack_size == 2) {
    BroadcastMulKernel<T, IndexType, 2>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(x, y, out, cols,
                                                                                    elem_cnt);
  } else {
    BroadcastMulKernel<T, IndexType, 1>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(x, y, out, cols,
                                                                                    elem_cnt);
  }
}

template<typename T>
void DispatchBroadcastMulIndexType(ep::Stream* stream, const T* x, const T* y, T* out,
                                   const int64_t cols, const int64_t elem_cnt) {
  if (elem_cnt < GetMaxVal<int32_t>()) {
    DispatchBroadcastMulPackSize<T, int32_t>(stream, x, y, out, cols, elem_cnt);
  } else {
    DispatchBroadcastMulPackSize<T, int64_t>(stream, x, y, out, cols, elem_cnt);
  }
}

template<typename T, typename IndexType, int pack_size>
__global__ void BroadcastAddElementwiseMulKernel(const T* x, const T* y, const T* z, T* out,
                                                 const IndexType cols, const IndexType elem_cnt) {
  const IndexType global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  using LoadPack = cuda::elementwise::Packed<T, pack_size>;
  for (IndexType linear_index = global_thread_id * pack_size,
                 step = gridDim.x * blockDim.x * pack_size;
       linear_index < elem_cnt; linear_index += step) {
    const IndexType row_idx = linear_index / cols;
    const IndexType col_idx = linear_index - row_idx * cols;
    const LoadPack* x_load = reinterpret_cast<const LoadPack*>(x + linear_index);
    const LoadPack* y_load = reinterpret_cast<const LoadPack*>(y + col_idx);
    const LoadPack* z_load = reinterpret_cast<const LoadPack*>(z + linear_index);

    LoadPack x_vec = *x_load;
    LoadPack y_vec = *y_load;
    LoadPack z_vec = *z_load;
    LoadPack out_store;

#pragma unroll
    for (int i = 0; i < pack_size; i++) {
      out_store.elem[i] = (x_vec.elem[i] + y_vec.elem[i]) * z_vec.elem[i];
    }
    *(reinterpret_cast<LoadPack*>(out + linear_index)) = out_store;
  }
}

template<typename T, typename IndexType>
void DispatchBroadcastAddElementwiseMulPackSize(ep::Stream* stream, const T* x, const T* y,
                                                const T* z, T* out, const IndexType cols,
                                                const IndexType elem_cnt) {
  int grid_size;
  const int pack_size = GetLaunchPackSize<T>(cols);
  const int64_t pack_num = elem_cnt / pack_size;
  cudaError_t err = cuda::elementwise::GetNumBlocks(pack_num, &grid_size);
  if (pack_size == 8) {
    BroadcastAddElementwiseMulKernel<T, IndexType, 8>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(x, y, z, out,
                                                                                    cols, elem_cnt);
  } else if (pack_size == 4) {
    BroadcastAddElementwiseMulKernel<T, IndexType, 4>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(x, y, z, out,
                                                                                    cols, elem_cnt);
  } else if (pack_size == 2) {
    BroadcastAddElementwiseMulKernel<T, IndexType, 2>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(x, y, z, out,
                                                                                    cols, elem_cnt);
  } else {
    BroadcastAddElementwiseMulKernel<T, IndexType, 1>
        <<<grid_size, kBlockSize, 0, stream->As<ep::CudaStream>()->cuda_stream()>>>(x, y, z, out,
                                                                                    cols, elem_cnt);
  }
}

template<typename T>
void DispatchBroadcastAddElementwiseMulIndexType(ep::Stream* stream, const T* x, const T* y,
                                                 const T* z, T* out, const int64_t cols,
                                                 const int64_t elem_cnt) {
  if (elem_cnt < GetMaxVal<int32_t>()) {
    DispatchBroadcastAddElementwiseMulPackSize<T, int32_t>(stream, x, y, z, out, cols, elem_cnt);
  } else {
    DispatchBroadcastAddElementwiseMulPackSize<T, int64_t>(stream, x, y, z, out, cols, elem_cnt);
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
class FusedCrossFeatureInteractionGradKernel final : public OpKernel, public CudaGraphSupport {
 public:
  FusedCrossFeatureInteractionGradKernel() = default;
  ~FusedCrossFeatureInteractionGradKernel() override = default;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  using user_op::OpKernel::Compute;
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const Tensor* x0 = ctx->Tensor4ArgNameAndIndex("x0", 0);
    const Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const Tensor* matmul_result = ctx->Tensor4ArgNameAndIndex("matmul_result", 0);

    const int64_t batch_size = dy->shape_view().At(0);
    const int64_t hidden_size = dy->shape_view().At(1);
    const int64_t out_size = weight->shape_view().At(0);
    const int64_t dy_elem_cnt = dy->shape_view().elem_cnt();

    Tensor* dx0 = ctx->Tensor4ArgNameAndIndex("dx0", 0);
    Tensor* dw = ctx->Tensor4ArgNameAndIndex("dw", 0);
    Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    Tensor* dbias = ctx->Tensor4ArgNameAndIndex("dbias", 0);
    Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    // step1: Get dbias.
    const T* ones = nullptr;
    auto* cuda_device = dynamic_cast<ep::CudaDevice*>(ctx->stream()->device());
    if (cuda_device != nullptr) {
      ones = static_cast<const T*>(cuda_device->GetConstOnes(dy->data_type(), batch_size));
    }
    size_t m = 0, n = 0, k = 0;
    DimVector dy_shape(2);
    dy->shape_view().ToDimVector(&dy_shape);
    DimVector ones_buf_shape(2);
    ones_buf_shape.at(0) = 1;
    ones_buf_shape.at(1) = batch_size;
    InferMatmulMNK(ones_buf_shape, dy_shape, /*trans_a=*/false, /*trans_b=*/false, &m, &n, &k);
    auto reduce_matmul = NewReduceMatmulPrimitive(ctx);
    CHECK(reduce_matmul);
    reduce_matmul->Launch(ctx->stream(), m, n, k, 1.0, ones, dy->dptr(), 0.0, dbias->mut_dptr());

    // step2: Get dmatmul_result0.
    T* dy_mul_x0 = reinterpret_cast<T*>(tmp_buffer->mut_dptr());
    T* dmatmul_result0 = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>()
                                              + GetCudaAlignedSize(dy_elem_cnt * sizeof(T)));
    OF_CUDA_CHECK(cuda::elementwise::Binary(MulOp<T>(), dy_elem_cnt, dy_mul_x0, dy->dptr<T>(),
                                            x0->dptr<T>(),
                                            ctx->stream()->As<ep::CudaStream>()->cuda_stream()));

    ones = static_cast<const T*>(cuda_device->GetConstOnes(dy->data_type(), hidden_size));
    DimVector dy_mul_x0_shape(2);
    dy->shape_view().ToDimVector(&dy_mul_x0_shape);
    ones_buf_shape.at(0) = hidden_size;
    ones_buf_shape.at(1) = 1;
    InferMatmulMNK(dy_mul_x0_shape, ones_buf_shape, /*trans_a=*/false, /*trans_b=*/false, &m, &n,
                   &k);
    reduce_matmul->Launch(ctx->stream(), m, n, k, 1.0, dy_mul_x0, ones, 0.0, dmatmul_result0);

    // step3: Get dx
    T* dx_buf = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>()
                                     + GetCudaAlignedSize(dy_elem_cnt * sizeof(T))
                                     + GetCudaAlignedSize(batch_size * sizeof(T)));
    DimVector dmatmul_result_shape(2);
    dmatmul_result_shape.at(0) = batch_size;
    dmatmul_result_shape.at(1) = 1;  // todo change to hidden size
    DimVector weight_shape(2);
    weight->shape_view().ToDimVector(&weight_shape);
    InferMatmulMNK(dmatmul_result_shape, weight_shape, /*trans_a=*/false, /*trans_b=*/false, &m, &n,
                   &k);
    reduce_matmul->Launch(ctx->stream(), m, n, k, 1.0, dmatmul_result0, weight->dptr(), 0.0,
                          reinterpret_cast<void*>(dx_buf));
    OF_CUDA_CHECK(cuda::elementwise::Binary(AddOp<T>(), dy_elem_cnt, dx->mut_dptr<T>(), dx_buf,
                                            dy->dptr<T>(),
                                            ctx->stream()->As<ep::CudaStream>()->cuda_stream()));

    // step4: Get dw.
    DimVector x_shape(2);
    x->shape_view().ToDimVector(&x_shape);

    InferMatmulMNK(dmatmul_result_shape, x_shape, /*trans_a=*/true, /*trans_b=*/false, &m, &n, &k);
    auto weight_grad_matmul = NewWeightGradMatmulPrimitive(ctx);
    CHECK(weight_grad_matmul);
    weight_grad_matmul->Launch(ctx->stream(), m, n, k, 1.0, dmatmul_result0, x->dptr(), 0.0,
                               dw->mut_dptr());

    // step5: Get dx0.
    DispatchBroadcastMulIndexType<T>(ctx->stream(), dy->dptr<T>(), matmul_result->dptr<T>(),
                                     dx0->mut_dptr<T>(), hidden_size, dy_elem_cnt);
  }
};

#define REGISTER_FUSED_CROSS_FEATURE_INTERACTION_V1_GRAD_KERNEL(dtype)                        \
  REGISTER_USER_KERNEL("fused_cross_feature_interaction_v1_grad")                             \
      .SetCreateFn<FusedCrossFeatureInteractionGradKernel<dtype>>()                           \
      .SetIsMatchedHob((HobDeviceType() == DeviceType::kCUDA)                                 \
                       && (HobDataType("dy", 0) == GetDataType<dtype>::value)                 \
                       && ReduceMatmulPrimitiveExists() && WeightGradMatmulPrimitiveExists()) \
      .SetInferTmpSizeFn([](InferContext* ctx) {                                              \
        size_t tmp_size = 0;                                                                  \
        const TensorDesc& dy = ctx->InputTensorDesc("dy", 0);                                 \
        const int64_t dy_elem_cnt = dy.shape().elem_cnt();                                    \
        const int64_t batch_size = dy.shape().At(0);                                          \
        size_t dy_mul_x0_size = GetCudaAlignedSize(dy_elem_cnt * sizeof(dtype));              \
        size_t dmatmul_result_size = GetCudaAlignedSize(batch_size * sizeof(dtype));          \
        size_t dx_buf_size = dy_mul_x0_size;                                                  \
        tmp_size = dy_mul_x0_size + dmatmul_result_size + dx_buf_size;                        \
        return tmp_size;                                                                      \
      });

REGISTER_FUSED_CROSS_FEATURE_INTERACTION_V1_GRAD_KERNEL(float)
REGISTER_FUSED_CROSS_FEATURE_INTERACTION_V1_GRAD_KERNEL(half)

template<typename T>
class FusedCrossFeatureInteractionV2GradKernel final : public OpKernel, public CudaGraphSupport {
 public:
  FusedCrossFeatureInteractionV2GradKernel() = default;
  ~FusedCrossFeatureInteractionV2GradKernel() = default;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  using user_op::OpKernel::Compute;
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    const Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
    const Tensor* x0 = ctx->Tensor4ArgNameAndIndex("x0", 0);
    const Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const Tensor* matmul_result = ctx->Tensor4ArgNameAndIndex("matmul_result", 0);

    const int64_t batch_size = dy->shape_view().At(0);
    const int64_t in_size = weight->shape_view().At(1);
    const int64_t hidden_size = weight->shape_view().At(0);
    const int64_t dy_elem_cnt = dy->shape_view().elem_cnt();

    Tensor* dx0 = ctx->Tensor4ArgNameAndIndex("dx0", 0);
    Tensor* dw = ctx->Tensor4ArgNameAndIndex("dw", 0);
    Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    Tensor* dbias = ctx->Tensor4ArgNameAndIndex("dbias", 0);
    Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    // step1: Get dx0.
    DispatchBroadcastAddElementwiseMulIndexType<T>(ctx->stream(), matmul_result->dptr<T>(),
                                                   bias->dptr<T>(), dy->dptr<T>(),
                                                   dx0->mut_dptr<T>(), hidden_size, dy_elem_cnt);

    // step2: Get dmatmul_result0.
    T* dmatmul_result0 = reinterpret_cast<T*>(tmp_buffer->mut_dptr());
    OF_CUDA_CHECK(cuda::elementwise::Binary(MulOp<T>(), dy_elem_cnt, dmatmul_result0, dy->dptr<T>(),
                                            x0->dptr<T>(),
                                            ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
    // step3: Get dx
    T* dx_buf = reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>()
                                     + GetCudaAlignedSize(dy_elem_cnt * sizeof(T)));
    DimVector dmatmul_result_shape(2);
    dmatmul_result_shape.at(0) = batch_size;
    dmatmul_result_shape.at(1) = hidden_size;
    DimVector weight_shape(2);
    weight->shape_view().ToDimVector(&weight_shape);
    size_t m = 0, n = 0, k = 0;
    InferMatmulMNK(dmatmul_result_shape, weight_shape, /*trans_a=*/false, /*trans_b=*/false, &m, &n,
                   &k);
    auto reduce_matmul = NewReduceMatmulPrimitive(ctx);
    CHECK(reduce_matmul);
    reduce_matmul->Launch(ctx->stream(), m, n, k, 1.0, dmatmul_result0, weight->dptr(), 0.0,
                          reinterpret_cast<void*>(dx_buf));
    OF_CUDA_CHECK(cuda::elementwise::Binary(AddOp<T>(), dy_elem_cnt, dx->mut_dptr<T>(), dx_buf,
                                            dy->dptr<T>(),
                                            ctx->stream()->As<ep::CudaStream>()->cuda_stream()));

    // step4: Get dw.
    DimVector x_shape(2);
    x->shape_view().ToDimVector(&x_shape);

    InferMatmulMNK(dmatmul_result_shape, x_shape, /*trans_a=*/true, /*trans_b=*/false, &m, &n, &k);
    auto weight_grad_matmul = NewWeightGradMatmulPrimitive(ctx);
    CHECK(weight_grad_matmul);
    weight_grad_matmul->Launch(ctx->stream(), m, n, k, 1.0, dmatmul_result0, x->dptr(), 0.0,
                               dw->mut_dptr());

    // step5: Get dbias.
    const T* ones = nullptr;
    auto* cuda_device = dynamic_cast<ep::CudaDevice*>(ctx->stream()->device());
    if (cuda_device != nullptr) {
      ones = static_cast<const T*>(cuda_device->GetConstOnes(dy->data_type(), batch_size));
    }
    DimVector dy_shape(2);
    dy->shape_view().ToDimVector(&dy_shape);
    DimVector ones_buf_shape(2);
    ones_buf_shape.at(0) = 1;
    ones_buf_shape.at(1) = batch_size;
    InferMatmulMNK(ones_buf_shape, dy_shape, /*trans_a=*/false, /*trans_b=*/false, &m, &n, &k);
    reduce_matmul->Launch(ctx->stream(), m, n, k, 1.0, ones,
                          reinterpret_cast<void*>(dmatmul_result0), 0.0, dbias->mut_dptr());
  }
};

#define REGISTER_FUSED_CROSS_FEATURE_INTERACTION_V2_GRAD_KERNEL(dtype)                        \
  REGISTER_USER_KERNEL("fused_cross_feature_interaction_v2_grad")                             \
      .SetCreateFn<FusedCrossFeatureInteractionV2GradKernel<dtype>>()                         \
      .SetIsMatchedHob((HobDeviceType() == DeviceType::kCUDA)                                 \
                       && (HobDataType("dy", 0) == GetDataType<dtype>::value)                 \
                       && ReduceMatmulPrimitiveExists() && WeightGradMatmulPrimitiveExists()) \
      .SetInferTmpSizeFn([](InferContext* ctx) {                                              \
        size_t tmp_size = 0;                                                                  \
        const TensorDesc& dy = ctx->InputTensorDesc("dy", 0);                                 \
        const int64_t dy_elem_cnt = dy.shape().elem_cnt();                                    \
        size_t dmatmul_result_size = GetCudaAlignedSize(dy_elem_cnt * sizeof(dtype));         \
        size_t dx_buf_size = dmatmul_result_size;                                             \
        tmp_size = dmatmul_result_size + dx_buf_size;                                         \
        return tmp_size;                                                                      \
      });

REGISTER_FUSED_CROSS_FEATURE_INTERACTION_V2_GRAD_KERNEL(float)
REGISTER_FUSED_CROSS_FEATURE_INTERACTION_V2_GRAD_KERNEL(half)

}  // namespace user_op

}  // namespace oneflow
