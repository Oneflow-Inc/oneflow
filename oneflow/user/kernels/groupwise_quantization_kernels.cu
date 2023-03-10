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
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

template<typename T, int pack_size>
struct alignas(sizeof(T) * pack_size) AlignedArray {
  __device__ AlignedArray() {
    // do nothing
  }
  union {
    T elem[pack_size];
  };
};

template<typename Src, typename Dst, size_t pack_size>
struct Cast {
  __device__ void operator()(const AlignedArray<Src, pack_size>& src,
                             AlignedArray<Dst, pack_size>* dst) {
#pragma unroll
    for (int i = 0; i < pack_size; ++i) { dst->elem[i] = static_cast<Dst>(src.elem[i]); }
  }
};

template<typename Dst, size_t pack_size>
struct Cast<uint8_t, Dst, pack_size> {
  __device__ void operator()(const AlignedArray<uint8_t, pack_size>& src,
                             AlignedArray<Dst, pack_size>* dst) {
#pragma unroll
    for (int i = 0; i < pack_size; ++i) { dst->elem[i] = static_cast<Dst>(src.elem[i]); }
  }

  __device__ void operator()(const AlignedArray<uint8_t, pack_size>& src,
                             AlignedArray<Dst, pack_size * 2>* dst) {
#pragma unroll
    for (int i = 0; i < pack_size; ++i) {
      const uint8_t q = src.elem[i];
      const uint8_t hi = (q >> 4);
      const uint8_t lo = (q & 0xF);
      dst->elem[i * 2 + 0] = static_cast<Dst>(hi);
      dst->elem[i * 2 + 1] = static_cast<Dst>(lo);
    }
  }
};

template<typename Dst, size_t pack_size>
struct Cast<int8_t, Dst, pack_size> {
  __device__ void operator()(const AlignedArray<int8_t, pack_size>& src,
                             AlignedArray<Dst, pack_size>* dst) {
#pragma unroll
    for (int i = 0; i < pack_size; ++i) { dst->elem[i] = static_cast<Dst>(src.elem[i]); }
  }

  __device__ void operator()(const AlignedArray<int8_t, pack_size>& src,
                             AlignedArray<Dst, pack_size * 2>* dst) {
#pragma unroll
    for (int i = 0; i < pack_size; ++i) {
      const int8_t q = src.elem[i];
      const int8_t hi = (q >> 4);
      int8_t lo = (q << 4);
      lo = (lo >> 4);
      dst->elem[i * 2 + 0] = static_cast<Dst>(hi);
      dst->elem[i * 2 + 1] = static_cast<Dst>(lo);
    }
  }
};

template<typename C, size_t pack_size>
struct InplaceAddScalar {
  __device__ void operator()(AlignedArray<C, pack_size>* array, C scalar) {
#pragma unroll
    for (int i = 0; i < pack_size; ++i) { array->elem[i] += scalar; }
  }
};

template<typename T, size_t pack_size>
struct InplaceFmaScalar {
  __device__ void operator()(AlignedArray<T, pack_size>* array, T m, T a) {
#pragma unroll
    for (int i = 0; i < pack_size; ++i) { array->elem[i] = array->elem[i] * m + a; }
  }
};

#if __CUDA_ARCH_ >= 530
template<size_t pack_size>
struct InplaceFmaScalar<half, pack_size> {
  __device__ void operator()(AlignedArray<half, pack_size>* array, half m, half a) {
    if (pack_size == 1) {
#pragma unroll
      for (int i = 0; i < pack_size; ++i) { array->elem[i] = array->elem[i] * m + a; }
    } else {
      const half2 m2 = __half2half2(m);
      const half2 a2 = __half2half2(a);
      half2* h2 = reinterpret_cast<half2*>(array->elem);
#pragma unroll
      for (int i = 0; i < pack_size / 2; ++i) { h2[i] = __hfma2(h2[i], m2, a2); }
    }
  }
};
#endif  // __CUDA_ARCH_ >= 530

template<typename T, size_t pack_size>
struct InplaceFma {
  __device__ void operator()(AlignedArray<T, pack_size>* a, const AlignedArray<T, pack_size>& b,
                             const AlignedArray<T, pack_size>& c) {
#pragma unroll
    for (int i = 0; i < pack_size; ++i) { a->elem[i] = a->elem[i] * b.elem[i] + c.elem[i]; }
  }
};

template<typename T, size_t pack_size>
struct InplaceMulScalar {
  __device__ void operator()(AlignedArray<T, pack_size>* a, T b) {
#pragma unroll
    for (int i = 0; i < pack_size; ++i) { a->elem[i] = a->elem[i] * b; }
  }
};

template<typename T, typename C, size_t pack_size>
struct MultiplyAccumulate {
  __device__ void operator()(const AlignedArray<T, pack_size>& a,
                             const AlignedArray<T, pack_size>& b, C* sum) {
#pragma unroll
    for (int i = 0; i < pack_size; ++i) { *sum += static_cast<C>(a.elem[i] * b.elem[i]); }
  }
};

template<size_t pack_size>
struct MultiplyAccumulate<half, float, pack_size> {
  __device__ void operator()(const AlignedArray<half, pack_size>& a,
                             const AlignedArray<half, pack_size>& b, float* sum) {
    if (pack_size == 1) {
#pragma unroll
      for (int i = 0; i < pack_size; ++i) { *sum += static_cast<float>(a.elem[i] * b.elem[i]); }
    } else {
      const half2* a2 = reinterpret_cast<const half2*>(a.elem);
      const half2* b2 = reinterpret_cast<const half2*>(b.elem);

      for (int i = 0; i < pack_size / 2; ++i) {
        const half2 c2 = __hmul2(a2[i], b2[i]);
        const float2 f2 = __half22float2(c2);
        *sum += f2.x;
        *sum += f2.y;
      }
    }
  }
};

template<typename T, typename U, typename Index, size_t d_pack_size, size_t q_pack_size, int bits,
         bool symmetric, bool outer_size_1>
__global__ void Dequantize3D(Index packed_elem_cnt, Index group_size, Index packed_inner_size,
                             const AlignedArray<U, q_pack_size>* quantized,
                             const AlignedArray<T, d_pack_size>* scale,
                             const AlignedArray<T, d_pack_size>* zero,
                             AlignedArray<T, d_pack_size>* out) {
  const Index packed_group_inner_size = group_size * packed_inner_size;
  CUDA_1D_KERNEL_LOOP_T(Index, i, packed_elem_cnt) {
    const Index outer_id = outer_size_1 ? 0 : i / packed_group_inner_size;
    const Index group_inner_offset = i - outer_id * packed_group_inner_size;
    const Index group_id = group_inner_offset / packed_inner_size;
    const Index inner_id = group_inner_offset - group_id * packed_inner_size;
    const Index scale_offset = outer_id * packed_inner_size + inner_id;
    const AlignedArray<T, d_pack_size> group_scale = scale[scale_offset];
    AlignedArray<T, d_pack_size> group_zero;
    if (symmetric) {
      if (std::is_same<U, uint8_t>::value) {
        group_zero = group_scale;
        InplaceMulScalar<T, d_pack_size>()(&group_zero, -static_cast<T>(((1 << (bits - 1)) - 1)));
      } else {
#pragma unroll
        for (int i = 0; i < d_pack_size; ++i) { group_zero.elem[i] = 0; }
      }
    } else {
      group_zero = zero[scale_offset];
    }
    AlignedArray<T, d_pack_size> values;
    const AlignedArray<U, q_pack_size> q = quantized[i];
    Cast<U, T, q_pack_size>()(q, &values);
    InplaceFma<T, d_pack_size>()(&values, group_scale, group_zero);
    out[i] = values;
  }
}

template<typename T, typename U, int num_bits, bool symmetric, size_t d_pack_size,
         size_t q_pack_size, bool outer_size_1>
void LaunchDequantize3D(ep::CudaStream* stream, int64_t outer_size, int64_t group_size,
                        int64_t inner_size, const U* in, const T* scale, const T* zero, T* out) {
  if constexpr (sizeof(T) * d_pack_size <= 16 && q_pack_size > 0) {
    const int64_t packed_elem_cnt = outer_size * group_size * inner_size / d_pack_size;
    const int64_t packed_inner_size = inner_size / d_pack_size;
    if (packed_elem_cnt <= (1 << 30)) {
      RUN_CUDA_KERNEL((Dequantize3D<T, U, int32_t, d_pack_size, q_pack_size, num_bits, symmetric,
                                    outer_size_1>),
                      stream, packed_elem_cnt, packed_elem_cnt, group_size, packed_inner_size,
                      reinterpret_cast<const AlignedArray<U, q_pack_size>*>(in),
                      reinterpret_cast<const AlignedArray<T, d_pack_size>*>(scale),
                      reinterpret_cast<const AlignedArray<T, d_pack_size>*>(zero),
                      reinterpret_cast<AlignedArray<T, d_pack_size>*>(out));
    } else {
      RUN_CUDA_KERNEL((Dequantize3D<T, U, int64_t, d_pack_size, q_pack_size, num_bits, symmetric,
                                    outer_size_1>),
                      stream, packed_elem_cnt, packed_elem_cnt, group_size, packed_inner_size,
                      reinterpret_cast<const AlignedArray<U, q_pack_size>*>(in),
                      reinterpret_cast<const AlignedArray<T, d_pack_size>*>(scale),
                      reinterpret_cast<const AlignedArray<T, d_pack_size>*>(zero),
                      reinterpret_cast<AlignedArray<T, d_pack_size>*>(out));
    }
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T, typename U, int num_bits, bool symmetric, size_t d_pack_size,
         size_t q_pack_size>
void DispatchDequantize3DOuterSize1(ep::CudaStream* stream, int64_t outer_size, int64_t group_size,
                                    int64_t inner_size, const U* in, const T* scale, const T* zero,
                                    T* out) {
  if (outer_size == 1) {
    LaunchDequantize3D<T, U, num_bits, symmetric, d_pack_size, q_pack_size, true>(
        stream, outer_size, group_size, inner_size, in, scale, zero, out);
  } else {
    LaunchDequantize3D<T, U, num_bits, symmetric, d_pack_size, q_pack_size, false>(
        stream, outer_size, group_size, inner_size, in, scale, zero, out);
  }
}

template<typename T, typename U, int num_bits, bool symmetric>
void DispatchDequantize3D(ep::CudaStream* stream, int64_t outer_size, int64_t group_size,
                          int64_t inner_size, const U* in, const T* scale, const T* zero, T* out) {
  constexpr int32_t max_pack_size = 16 / sizeof(T);
  constexpr int32_t data_per_quant = 8 / num_bits;
  int32_t pack_size = max_pack_size;
  while (inner_size % pack_size != 0) { pack_size /= 2; }
  if (pack_size == 16) {
    DispatchDequantize3DOuterSize1<T, U, num_bits, symmetric, 16, 16 / data_per_quant>(
        stream, outer_size, group_size, inner_size, in, scale, zero, out);
  } else if (pack_size == 8) {
    DispatchDequantize3DOuterSize1<T, U, num_bits, symmetric, 8, 8 / data_per_quant>(
        stream, outer_size, group_size, inner_size, in, scale, zero, out);
  } else if (pack_size == 4) {
    DispatchDequantize3DOuterSize1<T, U, num_bits, symmetric, 4, 4 / data_per_quant>(
        stream, outer_size, group_size, inner_size, in, scale, zero, out);
  } else if (pack_size == 2) {
    DispatchDequantize3DOuterSize1<T, U, num_bits, symmetric, 2, 2 / data_per_quant>(
        stream, outer_size, group_size, inner_size, in, scale, zero, out);
  } else if (pack_size == 1) {
    DispatchDequantize3DOuterSize1<T, U, num_bits, symmetric, 1, 1 / data_per_quant>(
        stream, outer_size, group_size, inner_size, in, scale, zero, out);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T, typename U, typename Index, size_t d_pack_size, size_t q_pack_size, int bits,
         bool symmetric>
__global__ void DequantizeInnerSize1(Index packed_elem_cnt, Index packed_group_size,
                                     const AlignedArray<U, q_pack_size>* quantized, const T* scale,
                                     const T* zero, AlignedArray<T, d_pack_size>* out) {
  CUDA_1D_KERNEL_LOOP_T(Index, i, packed_elem_cnt) {
    const Index group_id = i / packed_group_size;
    const T group_scale = scale[group_id];
    T group_zero;
    if (symmetric) {
      if (std::is_same<U, uint8_t>::value) {
        group_zero = -static_cast<T>(((1 << (bits - 1)) - 1)) * group_scale;
      } else {
        group_zero = 0;
      }
    } else {
      group_zero = zero[group_id];
    }
    AlignedArray<T, d_pack_size> values;
    AlignedArray<U, q_pack_size> q = quantized[i];
    Cast<U, T, q_pack_size>()(q, &values);
    InplaceFmaScalar<T, d_pack_size>()(&values, group_scale, group_zero);
    out[i] = values;
  }
}

template<typename T, typename U, int num_bits, bool symmetric, size_t d_pack_size,
         size_t q_pack_size>
void LaunchDequantizeInnerSize1(ep::CudaStream* stream, int64_t outer_size, int64_t group_size,
                                const U* in, const T* scale, const T* zero, T* out) {
  if constexpr (sizeof(T) * d_pack_size <= 16 && q_pack_size > 0) {
    const int64_t packed_elem_cnt = outer_size * group_size / d_pack_size;
    const int64_t packed_group_size = group_size / d_pack_size;
    if (packed_elem_cnt <= (1 << 30)) {
      RUN_CUDA_KERNEL(
          (DequantizeInnerSize1<T, U, int32_t, d_pack_size, q_pack_size, num_bits, symmetric>),
          stream, packed_elem_cnt, packed_elem_cnt, packed_group_size,
          reinterpret_cast<const AlignedArray<U, q_pack_size>*>(in), scale, zero,
          reinterpret_cast<AlignedArray<T, d_pack_size>*>(out));
    } else {
      RUN_CUDA_KERNEL(
          (DequantizeInnerSize1<T, U, int64_t, d_pack_size, q_pack_size, num_bits, symmetric>),
          stream, packed_elem_cnt, packed_elem_cnt, packed_group_size,
          reinterpret_cast<const AlignedArray<U, q_pack_size>*>(in), scale, zero,
          reinterpret_cast<AlignedArray<T, d_pack_size>*>(out));
    }
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T, typename U, int num_bits, bool symmetric>
void DispatchDequantizeInnerSize1PackSize(ep::CudaStream* stream, int64_t outer_size,
                                          int64_t group_size, const U* in, const T* scale,
                                          const T* zero, T* out) {
  constexpr int32_t max_pack_size = 16 / sizeof(T);
  int32_t pack_size = max_pack_size;
  while (group_size % pack_size != 0) { pack_size /= 2; }
  constexpr int32_t data_per_quant = 8 / num_bits;
  CHECK(group_size % data_per_quant == 0);
  if (pack_size == 16) {
    LaunchDequantizeInnerSize1<T, U, num_bits, symmetric, 16, 16 / data_per_quant>(
        stream, outer_size, group_size, in, scale, zero, out);
  } else if (pack_size == 8) {
    LaunchDequantizeInnerSize1<T, U, num_bits, symmetric, 8, 8 / data_per_quant>(
        stream, outer_size, group_size, in, scale, zero, out);
  } else if (pack_size == 4) {
    LaunchDequantizeInnerSize1<T, U, num_bits, symmetric, 4, 4 / data_per_quant>(
        stream, outer_size, group_size, in, scale, zero, out);
  } else if (pack_size == 2) {
    LaunchDequantizeInnerSize1<T, U, num_bits, symmetric, 2, 2 / data_per_quant>(
        stream, outer_size, group_size, in, scale, zero, out);
  } else if (pack_size == 1) {
    LaunchDequantizeInnerSize1<T, U, num_bits, symmetric, 1, 1 / data_per_quant>(
        stream, outer_size, group_size, in, scale, zero, out);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T, typename U, int num_bits, bool symmetric>
void DispatchDequantizeSize(ep::CudaStream* stream, int64_t outer_size, int64_t group_size,
                            int64_t inner_size, const U* in, const T* scale, const T* zero,
                            T* out) {
  if (inner_size == 1) {
    DispatchDequantizeInnerSize1PackSize<T, U, num_bits, symmetric>(stream, outer_size, group_size,
                                                                    in, scale, zero, out);
  } else {
    DispatchDequantize3D<T, U, num_bits, symmetric>(stream, outer_size, group_size, inner_size, in,
                                                    scale, zero, out);
  }
}

template<typename T, typename U>
void DispatchDequantize(ep::CudaStream* stream, int32_t num_bits, bool symmetric,
                        int64_t outer_size, int64_t group_size, int64_t inner_size, const U* in,
                        const T* scale, const T* zero, T* out) {
  if (num_bits == 4) {
    if (symmetric) {
      DispatchDequantizeSize<T, U, 4, true>(stream, outer_size, group_size, inner_size, in, scale,
                                            zero, out);
    } else {
      DispatchDequantizeSize<T, U, 4, false>(stream, outer_size, group_size, inner_size, in, scale,
                                             zero, out);
    }
  } else if (num_bits == 8) {
    if (symmetric) {
      DispatchDequantizeSize<T, U, 8, true>(stream, outer_size, group_size, inner_size, in, scale,
                                            zero, out);
    } else {
      DispatchDequantizeSize<T, U, 8, false>(stream, outer_size, group_size, inner_size, in, scale,
                                             zero, out);
    }

  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
class GroupwiseDequantizeKernel final : public user_op::OpKernel {
 public:
  GroupwiseDequantizeKernel() = default;
  ~GroupwiseDequantizeKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* scale = ctx->Tensor4ArgNameAndIndex("scale", 0);
    const user_op::Tensor* zero = nullptr;
    if (ctx->has_input("zero", 0)) { zero = ctx->Tensor4ArgNameAndIndex("zero", 0); }
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t group_size = ctx->Attr<int64_t>("group_size");
    const int64_t group_dim = ctx->Attr<int64_t>("group_dim");
    const int32_t num_bits = ctx->Attr<int32_t>("num_bits");
    const bool symmetric = ctx->Attr<bool>("symmetric");
    const int64_t num_in_axes = in->shape_view().NumAxes();
    CHECK_GE(num_in_axes, 1);
    CHECK_EQ(scale->shape_view().NumAxes(), num_in_axes);
    if (zero != nullptr) { CHECK_EQ(zero->shape_view().NumAxes(), num_in_axes); }
    CHECK_EQ(out->shape_view().NumAxes(), num_in_axes);
    CHECK_GE(group_dim, 0);
    CHECK_LT(group_dim, num_in_axes);
    for (int i = 0; i < num_in_axes; ++i) {
      if (i == num_in_axes - 1) {
        CHECK_EQ(out->shape_view().At(i), in->shape_view().At(i) * (8 / num_bits));
      } else {
        CHECK_EQ(out->shape_view().At(i), in->shape_view().At(i));
      }
    }
    const int64_t group_dim_size = out->shape_view().At(group_dim);
    CHECK_GT(group_size, 0);
    CHECK_LE(group_size, group_dim_size);
    CHECK_EQ(group_dim_size % group_size, 0);
    const int64_t num_groups = group_dim_size / group_size;
    for (int i = 0; i < num_in_axes; ++i) {
      const int64_t expected_dim_size = i == group_dim ? num_groups : out->shape_view().At(i);
      CHECK_EQ(scale->shape_view().At(i), expected_dim_size);
      if (zero != nullptr) { CHECK_EQ(zero->shape_view().At(i), expected_dim_size); }
    }
    const int64_t outer_size = out->shape_view().Count(0, group_dim) * num_groups;
    const int64_t inner_size = out->shape_view().Count(group_dim + 1);
    if (in->data_type() == DataType::kUInt8) {
      DispatchDequantize<T, uint8_t>(ctx->stream()->As<ep::CudaStream>(), num_bits, symmetric,
                                     outer_size, group_size, inner_size, in->dptr<uint8_t>(),
                                     scale->dptr<T>(), zero == nullptr ? nullptr : zero->dptr<T>(),
                                     out->mut_dptr<T>());
    } else if (in->data_type() == DataType::kInt8) {
      DispatchDequantize<T, int8_t>(ctx->stream()->As<ep::CudaStream>(), num_bits, symmetric,
                                    outer_size, group_size, inner_size, in->dptr<int8_t>(),
                                    scale->dptr<T>(), zero == nullptr ? nullptr : zero->dptr<T>(),
                                    out->mut_dptr<T>());
    } else {
      UNIMPLEMENTED();
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_VECTOR_WISE_SYMMETRIC_DEQUANTIZE_KERNEL(dtype)        \
  REGISTER_USER_KERNEL("groupwise_dequantize")                         \
      .SetCreateFn<GroupwiseDequantizeKernel<dtype>>()                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("scale", 0) == GetDataType<dtype>::value))

REGISTER_VECTOR_WISE_SYMMETRIC_DEQUANTIZE_KERNEL(half);
REGISTER_VECTOR_WISE_SYMMETRIC_DEQUANTIZE_KERNEL(float);

template<typename T, typename C, typename U, int block_size, size_t d_pack_size, size_t q_pack_size,
         int bits, bool symmetric, bool single_group>
__global__ void QuantizedMatmulBiasGroupN(int32_t M, int32_t N, int32_t K, int32_t group_size,
                                          const AlignedArray<T, d_pack_size>* __restrict__ x,
                                          const AlignedArray<U, q_pack_size>* __restrict__ w,
                                          const AlignedArray<T, d_pack_size>* __restrict__ scale,
                                          const AlignedArray<T, d_pack_size>* __restrict__ zero,
                                          const T* __restrict__ bias, T* __restrict__ out) {
  for (int32_t m = blockIdx.x; m < M; m += gridDim.x) {
    const auto* x_m = x + m * K;
    for (int32_t n = blockIdx.y; n < N; n += gridDim.y) {
      C t_sum = 0;
      const auto* w_n = w + n * K;
      const int64_t group_id = single_group ? 0 : n / group_size;
      const auto* scale_n = scale + group_id * K;
      const auto* zero_n = symmetric ? nullptr : zero + group_id * K;
      for (int32_t k = threadIdx.x; k < K; k += block_size) {
        auto xs = x_m[k];
        auto ws = w_n[k];
        auto scale_k = scale_n[k];
        AlignedArray<T, d_pack_size> zero_k;
        if (symmetric) {
          if (std::is_same<U, uint8_t>::value) {
            zero_k = scale_k;
            InplaceMulScalar<T, d_pack_size>()(&zero_k, -static_cast<T>(((1 << (bits - 1)) - 1)));
          } else {
            for (int i = 0; i < d_pack_size; ++i) { zero_k.elem[i] = 0; }
          }
        } else {
          zero_k = zero_n[k];
        }
        AlignedArray<T, d_pack_size> weights;
        Cast<U, T, q_pack_size>()(ws, &weights);
        InplaceFma<T, d_pack_size>()(&weights, scale_k, zero_k);
        MultiplyAccumulate<T, C, d_pack_size>()(xs, weights, &t_sum);
      }
      using BlockReduce = cub::BlockReduce<C, block_size>;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      C sum = BlockReduce(temp_storage).Sum(t_sum);
      if (threadIdx.x == 0) {
        if (bias != nullptr) { sum += static_cast<C>(bias[n]); }
        out[m * N + n] = static_cast<T>(sum);
      }
      __syncthreads();
    }
  }
}

template<typename T, typename C, typename U, int num_bits, bool symmetric, size_t d_pack_size,
         size_t q_pack_size, bool single_group>
void LaunchMatmulBiasGroupN(ep::CudaStream* stream, int64_t m, int64_t n, int64_t k,
                            int64_t group_size, const T* x, const U* w, const T* scale,
                            const T* zero, const T* bias, T* out) {
  constexpr uint32_t max_grid_size = 8192;
  constexpr uint32_t block_size = 128;
  const int64_t int32_max = std::numeric_limits<int32_t>::max();
  if (m * k > int32_max || n * k > int32_max || m * n > int32_max || m > int32_max - max_grid_size
      || n > int32_max - max_grid_size || k > int32_max - block_size) {
    UNIMPLEMENTED();
  }
  if constexpr (sizeof(T) * d_pack_size <= 16 && q_pack_size > 0) {
    QuantizedMatmulBiasGroupN<T, C, U, block_size, d_pack_size, q_pack_size, num_bits, symmetric,
                              single_group>
        <<<dim3(std::min<int64_t>(m, max_grid_size), std::min<int64_t>(n, max_grid_size)),
           block_size, 0, stream->cuda_stream()>>>(
            m, n, k / d_pack_size, group_size,
            reinterpret_cast<const AlignedArray<T, d_pack_size>*>(x),
            reinterpret_cast<const AlignedArray<U, q_pack_size>*>(w),
            reinterpret_cast<const AlignedArray<T, d_pack_size>*>(scale),
            reinterpret_cast<const AlignedArray<T, d_pack_size>*>(zero), bias, out);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T, typename C, typename U, int num_bits, bool symmetric, size_t d_pack_size,
         size_t q_pack_size>
void DispatchMatmulBiasGroupNSingleGroup(ep::CudaStream* stream, int64_t m, int64_t n, int64_t k,
                                         int64_t group_size, const T* x, const U* w, const T* scale,
                                         const T* zero, const T* bias, T* out) {
  if (n == group_size) {
    LaunchMatmulBiasGroupN<T, C, U, num_bits, symmetric, d_pack_size, q_pack_size, true>(
        stream, m, n, k, group_size, x, w, scale, zero, bias, out);
  } else {
    LaunchMatmulBiasGroupN<T, C, U, num_bits, symmetric, d_pack_size, q_pack_size, false>(
        stream, m, n, k, group_size, x, w, scale, zero, bias, out);
  }
}

template<typename T, typename C, typename U, int num_bits, bool symmetric>
void DispatchMatmulBiasGroupNPackSize(ep::CudaStream* stream, int64_t m, int64_t n, int64_t k,
                                      int64_t group_size, const T* x, const U* w, const T* scale,
                                      const T* zero, const T* bias, T* out) {
  const int max_pack_size = 16 / sizeof(T);
  int pack_size = max_pack_size;
  while (k % pack_size != 0) { pack_size /= 2; }
  constexpr int32_t data_per_quant = 8 / num_bits;
  if (pack_size == 16) {
    DispatchMatmulBiasGroupNSingleGroup<T, C, U, num_bits, symmetric, 16, 16 / data_per_quant>(
        stream, m, n, k, group_size, x, w, scale, zero, bias, out);
  } else if (pack_size == 8) {
    DispatchMatmulBiasGroupNSingleGroup<T, C, U, num_bits, symmetric, 8, 8 / data_per_quant>(
        stream, m, n, k, group_size, x, w, scale, zero, bias, out);
  } else if (pack_size == 4) {
    DispatchMatmulBiasGroupNSingleGroup<T, C, U, num_bits, symmetric, 4, 4 / data_per_quant>(
        stream, m, n, k, group_size, x, w, scale, zero, bias, out);
  } else if (pack_size == 2) {
    DispatchMatmulBiasGroupNSingleGroup<T, C, U, num_bits, symmetric, 2, 2 / data_per_quant>(
        stream, m, n, k, group_size, x, w, scale, zero, bias, out);
  } else if (pack_size == 1) {
    DispatchMatmulBiasGroupNSingleGroup<T, C, U, num_bits, symmetric, 1, 1 / data_per_quant>(
        stream, m, n, k, group_size, x, w, scale, zero, bias, out);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T, typename C, typename U, int block_size, size_t d_pack_size, size_t q_pack_size,
         int bits, bool symmetric, bool single_group>
__global__ void QuantizedMatmulBiasGroupK(int32_t M, int32_t N, int32_t K, int32_t group_size,
                                          int32_t num_groups_per_n,
                                          const AlignedArray<T, d_pack_size>* __restrict__ x,
                                          const AlignedArray<U, q_pack_size>* __restrict__ w,
                                          const T* __restrict__ scale, const T* __restrict__ zero,
                                          const T* __restrict__ bias, T* __restrict__ out) {
  for (int32_t m = blockIdx.x; m < M; m += gridDim.x) {
    const auto* x_m = x + m * K;
    for (int32_t n = blockIdx.y; n < N; n += gridDim.y) {
      C t_sum = 0;
      const auto* w_n = w + n * K;
      const auto* scale_n = scale + n * num_groups_per_n;
      const T* zero_n = symmetric ? nullptr : zero + n * num_groups_per_n;
      T group_scale;
      T group_zero;
      if (single_group) {
        group_scale = static_cast<T>(scale_n[0]);
        if (symmetric) {
          if (std::is_same<U, uint8_t>::value) {
            group_zero = -static_cast<T>(((1 << (bits - 1)) - 1)) * group_scale;
          } else {
            group_zero = 0;
          }
        } else {
          group_zero = zero_n[0];
        }
      }
      for (int32_t k = threadIdx.x; k < K; k += block_size) {
        if (!single_group) {
          auto group_id = k / group_size;
          group_scale = static_cast<T>(scale_n[group_id]);
          if (symmetric) {
            if (std::is_same<U, uint8_t>::value) {
              group_zero = -static_cast<T>(((1 << (bits - 1)) - 1)) * group_scale;
            } else {
              group_zero = 0;
            }
          } else {
            group_zero = zero_n[group_id];
          }
        }
        auto xs = x_m[k];
        auto ws = w_n[k];
        AlignedArray<T, d_pack_size> weights;
        Cast<U, T, q_pack_size>()(ws, &weights);
        InplaceFmaScalar<T, d_pack_size>()(&weights, group_scale, group_zero);
        MultiplyAccumulate<T, C, d_pack_size>()(xs, weights, &t_sum);
      }
      using BlockReduce = cub::BlockReduce<C, block_size>;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      C sum = BlockReduce(temp_storage).Sum(t_sum);
      if (threadIdx.x == 0) {
        if (bias != nullptr) { sum += static_cast<C>(bias[n]); }
        out[m * N + n] = static_cast<T>(sum);
      }
      __syncthreads();
    }
  }
}

template<typename T, typename C, typename U, int num_bits, bool symmetric, size_t d_pack_size,
         size_t q_pack_size, bool single_group>
void LaunchMatmulBiasGroupK(ep::CudaStream* stream, int64_t m, int64_t n, int64_t k,
                            int64_t group_size, const T* x, const U* w, const T* scale,
                            const T* zero, const T* bias, T* out) {
  constexpr uint32_t max_grid_size = 8192;
  constexpr uint32_t block_size = 128;
  const int64_t int32_max = std::numeric_limits<int32_t>::max();
  if (m * k > int32_max || n * k > int32_max || m * n > int32_max || m > int32_max - max_grid_size
      || n > int32_max - max_grid_size || k > int32_max - block_size) {
    UNIMPLEMENTED();
  }
  if constexpr (sizeof(T) * d_pack_size <= 16 && q_pack_size > 0) {
    QuantizedMatmulBiasGroupK<T, C, U, block_size, d_pack_size, q_pack_size, num_bits, symmetric,
                              single_group>
        <<<dim3(std::min<int64_t>(m, max_grid_size), std::min<int64_t>(n, max_grid_size)),
           block_size, 0, stream->cuda_stream()>>>(
            m, n, k / d_pack_size, group_size / d_pack_size, k / group_size,
            reinterpret_cast<const AlignedArray<T, d_pack_size>*>(x),
            reinterpret_cast<const AlignedArray<U, q_pack_size>*>(w), scale, zero, bias, out);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T, typename C, typename U, int num_bits, bool symmetric, size_t d_pack_size,
         size_t q_pack_size>
void DispatchMatmulBiasGroupKSingleGroup(ep::CudaStream* stream, int64_t m, int64_t n, int64_t k,
                                         int64_t group_size, const T* x, const U* w, const T* scale,
                                         const T* zero, const T* bias, T* out) {
  if (k == group_size) {
    LaunchMatmulBiasGroupK<T, C, U, num_bits, symmetric, d_pack_size, q_pack_size, true>(
        stream, m, n, k, group_size, x, w, scale, zero, bias, out);
  } else {
    LaunchMatmulBiasGroupK<T, C, U, num_bits, symmetric, d_pack_size, q_pack_size, false>(
        stream, m, n, k, group_size, x, w, scale, zero, bias, out);
  }
}

template<typename T, typename C, typename U, int num_bits, bool symmetric>
void DispatchMatmulBiasGroupKPackSize(ep::CudaStream* stream, int64_t m, int64_t n, int64_t k,
                                      int64_t group_size, const T* x, const U* w, const T* scale,
                                      const T* zero, const T* bias, T* out) {
  const int max_pack_size = 16 / sizeof(T);
  int pack_size = max_pack_size;
  while (group_size % pack_size != 0) { pack_size /= 2; }
  constexpr int32_t data_per_quant = 8 / num_bits;
  if (pack_size == 16) {
    DispatchMatmulBiasGroupKSingleGroup<T, C, U, num_bits, symmetric, 16, 16 / data_per_quant>(
        stream, m, n, k, group_size, x, w, scale, zero, bias, out);
  } else if (pack_size == 8) {
    DispatchMatmulBiasGroupKSingleGroup<T, C, U, num_bits, symmetric, 8, 8 / data_per_quant>(
        stream, m, n, k, group_size, x, w, scale, zero, bias, out);
  } else if (pack_size == 4) {
    DispatchMatmulBiasGroupKSingleGroup<T, C, U, num_bits, symmetric, 4, 4 / data_per_quant>(
        stream, m, n, k, group_size, x, w, scale, zero, bias, out);
  } else if (pack_size == 2) {
    DispatchMatmulBiasGroupKSingleGroup<T, C, U, num_bits, symmetric, 2, 2 / data_per_quant>(
        stream, m, n, k, group_size, x, w, scale, zero, bias, out);
  } else if (pack_size == 1) {
    DispatchMatmulBiasGroupKSingleGroup<T, C, U, num_bits, symmetric, 1, 1 / data_per_quant>(
        stream, m, n, k, group_size, x, w, scale, zero, bias, out);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T, typename C, typename U, int num_bits, bool symmetric>
void DispatchMatmulBiasGroupDim(ep::CudaStream* stream, int64_t m, int64_t n, int64_t k,
                                int64_t group_dim, int64_t group_size, const T* x, const U* w,
                                const T* scale, const T* zero, const T* bias, T* out) {
  if (group_dim == 0) {
    DispatchMatmulBiasGroupNPackSize<T, C, U, num_bits, symmetric>(stream, m, n, k, group_size, x,
                                                                   w, scale, zero, bias, out);
  } else if (group_dim == 1) {
    DispatchMatmulBiasGroupKPackSize<T, C, U, num_bits, symmetric>(stream, m, n, k, group_size, x,
                                                                   w, scale, zero, bias, out);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T, typename C, typename U>
void DispatchMatmulBias(ep::CudaStream* stream, int num_bits, bool symmetric, int64_t m, int64_t n,
                        int64_t k, int64_t group_dim, int64_t group_size, const T* x, const U* w,
                        const T* scale, const T* zero, const T* bias, T* out) {
  if (num_bits == 4) {
    if (symmetric) {
      DispatchMatmulBiasGroupDim<T, C, U, 4, true>(stream, m, n, k, group_dim, group_size, x, w,
                                                   scale, zero, bias, out);
    } else {
      DispatchMatmulBiasGroupDim<T, C, U, 4, false>(stream, m, n, k, group_dim, group_size, x, w,
                                                    scale, zero, bias, out);
    }
  } else if (num_bits == 8) {
    if (symmetric) {
      DispatchMatmulBiasGroupDim<T, C, U, 8, true>(stream, m, n, k, group_dim, group_size, x, w,
                                                   scale, zero, bias, out);
    } else {
      DispatchMatmulBiasGroupDim<T, C, U, 8, false>(stream, m, n, k, group_dim, group_size, x, w,
                                                    scale, zero, bias, out);
    }
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T>
class FusedLinearWithGroupwiseQuantizedWeightKernel final : public user_op::OpKernel,
                                                            public user_op::CudaGraphSupport {
 public:
  FusedLinearWithGroupwiseQuantizedWeightKernel() = default;
  ~FusedLinearWithGroupwiseQuantizedWeightKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* w = ctx->Tensor4ArgNameAndIndex("w", 0);
    const user_op::Tensor* w_scale = ctx->Tensor4ArgNameAndIndex("w_scale", 0);
    const user_op::Tensor* b =
        (ctx->has_input("b", 0)) ? ctx->Tensor4ArgNameAndIndex("b", 0) : nullptr;
    const user_op::Tensor* w_zero =
        (ctx->has_input("w_zero", 0)) ? ctx->Tensor4ArgNameAndIndex("w_zero", 0) : nullptr;
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const DataType data_type = x->data_type();
    CHECK_EQ(w_scale->data_type(), data_type);
    CHECK_EQ(out->data_type(), data_type);
    const int64_t group_size = ctx->Attr<int64_t>("group_size");
    const int64_t group_dim = ctx->Attr<int64_t>("group_dim");
    CHECK(group_dim == 0 || group_dim == 1);
    const int32_t num_bits = ctx->Attr<int32_t>("num_bits");
    const bool symmetric = ctx->Attr<bool>("symmetric");
    CHECK_GE(x->shape_view().NumAxes(), 2);
    const int64_t k = x->shape_view().At(x->shape_view().NumAxes() - 1);
    const int64_t m = x->shape_view().elem_cnt() / k;
    CHECK_EQ(w->shape_view().NumAxes(), 2);
    if (num_bits == 4) {
      CHECK_EQ(w->shape_view().At(1) * 2, k);
    } else if (num_bits == 8) {
      CHECK_EQ(w->shape_view().At(1), k);
    } else {
      UNIMPLEMENTED();
    }
    const int64_t n = w->shape_view().At(0);
    const int64_t group_dim_size = group_dim == 0 ? n : k;
    CHECK_GT(group_size, 0);
    CHECK_LE(group_size, group_dim_size);
    CHECK_EQ(group_dim_size % group_size, 0);
    const int64_t num_groups = group_dim_size / group_size;
    if (group_dim == 0) {
      CHECK_EQ(w_scale->shape_view().At(0), num_groups);
      CHECK_EQ(w_scale->shape_view().At(1), k);
    } else if (group_dim == 1) {
      CHECK_EQ(w_scale->shape_view().At(0), n);
      CHECK_EQ(w_scale->shape_view().At(1), num_groups);
    } else {
      UNIMPLEMENTED();
    }
    if (w_zero != nullptr) {
      CHECK_EQ(w_zero->data_type(), data_type);
      CHECK(w_zero->shape_view() == w_scale->shape_view());
    }
    if (b != nullptr) {
      CHECK_EQ(b->data_type(), data_type);
      CHECK_EQ(b->shape_view().NumAxes(), 1);
      CHECK_EQ(b->shape_view().At(0), n);
    }
    CHECK_EQ(x->shape_view().NumAxes(), out->shape_view().NumAxes());
    for (int i = 0; i < x->shape_view().NumAxes() - 1; ++i) {
      CHECK_EQ(out->shape_view().At(i), x->shape_view().At(i));
    }
    CHECK_EQ(out->shape_view().At(out->shape_view().NumAxes() - 1), n);
    if (symmetric) {
      CHECK(w_zero == nullptr);
    } else {
      CHECK(w_zero != nullptr);
    }
    const DataType quant_type = w->data_type();
    if (quant_type == DataType::kUInt8) {
      DispatchMatmulBias<T, float, uint8_t>(
          ctx->stream()->As<ep::CudaStream>(), num_bits, symmetric, m, n, k, group_dim, group_size,
          x->dptr<T>(), w->dptr<uint8_t>(), w_scale->dptr<T>(),
          w_zero == nullptr ? nullptr : w_zero->dptr<T>(), b == nullptr ? nullptr : b->dptr<T>(),
          out->mut_dptr<T>());
    } else if (quant_type == DataType::kInt8) {
      DispatchMatmulBias<T, float, int8_t>(
          ctx->stream()->As<ep::CudaStream>(), num_bits, symmetric, m, n, k, group_dim, group_size,
          x->dptr<T>(), w->dptr<int8_t>(), w_scale->dptr<T>(),
          w_zero == nullptr ? nullptr : w_zero->dptr<T>(), b == nullptr ? nullptr : b->dptr<T>(),
          out->mut_dptr<T>());
    } else {
      UNIMPLEMENTED();
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_MATMUL_BIAS_KERNEL_GPU(data_type, cpp_type)            \
  REGISTER_USER_KERNEL("fused_linear_with_groupwise_quantized_weight")        \
      .SetCreateFn<FusedLinearWithGroupwiseQuantizedWeightKernel<cpp_type>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)        \
                       && (user_op::HobDataType("out", 0) == data_type));

REGISTER_FUSED_MATMUL_BIAS_KERNEL_GPU(DataType::kFloat, float);
REGISTER_FUSED_MATMUL_BIAS_KERNEL_GPU(DataType::kFloat16, half);

}  // namespace

}  // namespace oneflow
