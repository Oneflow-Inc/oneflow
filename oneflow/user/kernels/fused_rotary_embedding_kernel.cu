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
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

void ParseDims(const ShapeView& shape, const std::string& layout,
               const Optional<int64_t>& num_heads, const Optional<int64_t>& head_size,
               int64_t tensor_index, int64_t* b, int64_t* m, int64_t* h, int64_t* k,
               int64_t* b_stride, int64_t* m_stride, int64_t* h_stride, int64_t* offset) {
  if (shape.NumAxes() == 3) {
    if (layout == "BM(HK)" || layout == "BM(H2K)" || layout == "BM(H3K)" || layout == "MB(HK)"
        || layout == "MB(H2K)" || layout == "MB(H3K)") {
      bool batch_first = false;
      int64_t packed_n = 0;
      const std::string layout_bm = layout.substr(0, 2);
      const std::string layout_hk = layout.substr(2);
      if (layout_bm == "BM") {
        *b = shape.At(0);
        *m = shape.At(1);
        batch_first = true;
      } else if (layout_bm == "MB") {
        *b = shape.At(1);
        *m = shape.At(0);
        batch_first = false;
      } else {
        UNIMPLEMENTED();
      }
      if (layout_hk == "(HK)") {
        packed_n = 1;
      } else if (layout_hk == "(H2K)") {
        packed_n = 2;
      } else if (layout_hk == "(H3K)") {
        packed_n = 3;
      } else {
        UNIMPLEMENTED();
      }
      const int64_t hidden_size = shape.At(2);
      if (num_heads) {
        const int64_t expected_h = CHECK_JUST(num_heads);
        const int64_t packed_h = packed_n * expected_h;
        CHECK_EQ(hidden_size % packed_h, 0);
        *h = expected_h;
        *k = hidden_size / packed_h;
      } else if (head_size) {
        const int64_t expected_k = CHECK_JUST(head_size);
        const int64_t packed_k = packed_n * expected_k;
        CHECK_EQ(hidden_size % packed_k, 0);
        *h = hidden_size / packed_k;
        *k = expected_k;
      } else {
        UNIMPLEMENTED();
      }
      *h_stride = *k * packed_n;
      if (batch_first) {
        *m_stride = *h_stride * *h;
        *b_stride = *m_stride * *m;
      } else {
        *b_stride = *h_stride * *h;
        *m_stride = *b_stride * *b;
      }
      if (packed_n == 1) {
        *offset = 0;
      } else if (packed_n == 2) {
        CHECK_GE(tensor_index, 1);
        *offset = (tensor_index - 1) * *k;
      } else if (packed_n == 3) {
        *offset = tensor_index * *k;
      } else {
        UNIMPLEMENTED();
      }
    } else {
      UNIMPLEMENTED();
    }
  } else if (shape.NumAxes() == 4) {
    if (layout == "BMHK") {
      *b = shape.At(0);
      *m = shape.At(1);
      *h = shape.At(2);
      *k = shape.At(3);
      *h_stride = *k;
      *m_stride = *h_stride * *h;
      *b_stride = *m_stride * *m;
    } else if (layout == "BHMK") {
      *b = shape.At(0);
      *m = shape.At(2);
      *h = shape.At(1);
      *k = shape.At(3);
      *m_stride = *k;
      *h_stride = *m_stride * *m;
      *b_stride = *h_stride * *h;
    } else if (layout == "MBHK") {
      *b = shape.At(1);
      *m = shape.At(0);
      *h = shape.At(2);
      *k = shape.At(3);
      *h_stride = *k;
      *b_stride = *h_stride * *h;
      *m_stride = *b_stride * *b;
    } else {
      UNIMPLEMENTED();
    }
    if (num_heads) {
      const int64_t expected_h = CHECK_JUST(num_heads);
      CHECK_EQ(*h, expected_h);
    }
    if (head_size) {
      const int64_t expected_k = CHECK_JUST(head_size);
      CHECK_EQ(*k, expected_k);
    }
    *offset = 0;
  } else {
    UNIMPLEMENTED();
  };
}

template<typename T, int num_dims>
struct FusedApplyRotaryEmbParam {
  const T* x;
  const T* cos;
  const T* sin;
  T* out;
  int32_t num_elements;
  int64_t k;
  int64_t offset;
  int64_t x_stride[num_dims];
  int64_t sinuous_stride[num_dims];
  int64_t sinuous_mask[num_dims];

  FusedApplyRotaryEmbParam(const T* x, const T* cos, const T* sin, T* out,
                                    const int32_t num_elements, const int64_t k, const int64_t offset)
      : x(x), cos(cos), sin(sin), out(out), num_elements(num_elements), k(k), offset(offset) {}
};

template<typename T, typename IndexType, size_t pack_size, size_t num_dims>
__global__ void FusedApplyRotaryEmbComputeKernel(FusedApplyRotaryEmbParam<T, num_dims> param) {
  const T* x = param.x;
  const T* cos = param.cos;
  const T* sin = param.sin;
  T* out = param.out;
  const IndexType num_elements = param.num_elements;
  for (IndexType packed_offset = threadIdx.x + blockIdx.x * blockDim.x; packed_offset < num_elements;
       packed_offset += blockDim.x * gridDim.x) {
    using LoadPack = cuda::elementwise::Packed<T, pack_size>;
    IndexType offset = param.offset + packed_offset * pack_size;
    const LoadPack* x_load = reinterpret_cast<const LoadPack*>(x + offset);
    const LoadPack x_vec = *x_load;
    IndexType m_index, k_index;
    IndexType sinuous_offset = 0;

#pragma unloop
    for (int i = 0; i < num_dims; i++) {
      IndexType index = offset / param.x_stride[i];
      offset = offset - index * param.x_stride[i];
      sinuous_offset = sinuous_offset + (index * param.sinuous_mask[i] * param.sinuous_stride[i]);
    }
    
    const LoadPack* cos_load = reinterpret_cast<const LoadPack*>(cos + sinuous_offset);
    const LoadPack* sin_load = reinterpret_cast<const LoadPack*>(sin + sinuous_offset);
    const LoadPack cos_vec = *cos_load, sin_vec = *sin_load;
    LoadPack out_vec;

#pragma unloop
    for (int i = 0; i < pack_size / 2; i++) {
      out_vec.elem[i * 2] =
          x_vec.elem[i * 2] * cos_vec.elem[i * 2] - x_vec.elem[i * 2 + 1] * sin_vec.elem[i * 2];
      out_vec.elem[i * 2 + 1] = x_vec.elem[i * 2 + 1] * cos_vec.elem[i * 2 + 1]
                                + x_vec.elem[i * 2] * sin_vec.elem[i * 2 + 1];
    }

    *(reinterpret_cast<LoadPack*>(out + param.offset + packed_offset * pack_size)) = out_vec;
  }
}

template<typename T, typename IndexType, size_t pack_size, size_t num_dims>
void LaunchKernel(const T* x, const T* cos, const T* sin, T* out, const std::string& layout, const int64_t b,
                    const int64_t m, const int64_t h, const int64_t k, const int64_t b_stride, 
                    const int64_t m_stride, const int64_t h_stride, const int64_t offset, 
                    IndexType num_elements) {
  DimVector kernel_x_shape(num_dims), kernel_sinuous_shape(num_dims);
  size_t x_stride[num_dims];
  size_t sinuous_stride[num_dims];

  x_stride[num_dims - 1] = 1;
  sinuous_stride[num_dims - 1] = 1;

  for (int i = num_dims - 2; i >= 0; i--) {
    x_stride[i] = x_stride[i + 1] * kernel_x_shape.at(i + 1);
    sinuous_stride[i] = sinuous_stride[i + 1] * kernel_sinuous_shape.at(i + 1);
  }

  struct FusedApplyRotaryEmbParam<T, num_dims> param(
      reinterpret_cast<const T*>(x), reinterpret_cast<const T*>(cos),
      reinterpret_cast<const T*>(sin), reinterpret_cast<T*>(out), num_elements, k, offset);

  std::pair<char, std::int64_t> strides[num_dims];
  strides[0] = {'b', b_stride};
  strides[1] = {'h', h_stride};
  strides[2] = {'m', m_stride};
  strides[3] = {'k', 1};

  auto GetDim = [&](const char c) {
    if (c == 'b') {
        return b;
    } else if (c == 'h') {
        return h;
    } else if (c == 'm') {
        return m;
    } else if (c == 'k') {
        return k;
    }

    return 0L;
  };

  std::sort(strides, strides + num_dims, [&](auto pair1, auto pair2) {
    if (pair1.second > pair2.second) {
        return true;
    } else if (pair1.second == pair2.second) {
        if (GetDim(pair1.first) != 1) {
            return true;
        }
        return false;
    } else {
        return false;
    }
    return pair1.second > pair2.second;
  });

// K has to be the last dimension, therefore sinuous_stride has to be [k, k, k, 1]
#pragma unloop
  for (int i = 0; i < num_dims; i++) {
    param.x_stride[i] = strides[i].second;
    param.sinuous_stride[i] = k;
    param.sinuous_mask[i] = 0;
    if (strides[i].first == 'm') {
      param.sinuous_mask[i] = 1;
    } else if (strides[i].first == 'k') {
      param.sinuous_mask[i] = 1;
      param.sinuous_stride[i] = 1;
    }
  }

  constexpr size_t blk_size = 128;
  FusedApplyRotaryEmbComputeKernel<T, IndexType, pack_size, num_dims>
      <<<(param.num_elements + blk_size - 1) / blk_size, blk_size>>>(param);
}

template<typename T, typename IndexType, size_t num_dims>
void DispatchPackSize(const T* x, const T* cos, const T* sin, T* out, const std::string& layout, const int64_t b, 
                      const int64_t m, const int64_t h, const int64_t k, const int64_t b_stride,
                      const int64_t m_stride, const int64_t h_stride, const int64_t offset, 
                      IndexType num_elements) {
  const auto CheckPackSize = [&](const size_t pack_size) {
    bool r = (((reinterpret_cast<uintptr_t>(x) % (sizeof(T) * pack_size)) == 0)
              && ((k % pack_size) == 0) && ((16 / sizeof(T)) >= pack_size));
    return r;
  };

  if (CheckPackSize(8)) {
    num_elements /= 8;
    LaunchKernel<T, IndexType, 8, num_dims>(x, cos, sin, out, layout, b, m, h, k, b_stride, m_stride, h_stride, offset,
                                            num_elements);
  } else if (CheckPackSize(4)) {
    num_elements /= 4;
    LaunchKernel<T, IndexType, 4, num_dims>(x, cos, sin, out, layout, b, m, h, k, b_stride, m_stride, h_stride, offset,
                                            num_elements);
  } else {
    num_elements /= 2;
    LaunchKernel<T, IndexType, 2, num_dims>(x, cos, sin, out, layout, b, m, h, k, b_stride, m_stride, h_stride, offset,
                                            num_elements);
  }
}

template<typename T, size_t num_dims>
void DispatchIndex(const T* x, const T* cos, const T* sin, T* out, const std::string& layout, const int64_t b, const int64_t m,
  const int64_t h, const int64_t k, const int64_t b_stride, const int64_t m_stride, const int64_t h_stride,
  const int64_t offset) {
  int64_t num_elements = b * m * h * k;
  if (num_elements < (1 << 30)) {
    DispatchPackSize<T, int32_t, num_dims>(x, cos, sin, out, layout, b, m, h, k, b_stride, m_stride, h_stride, offset,
                                           static_cast<int32_t>(num_elements));
  } else {
    DispatchPackSize<T, int64_t, num_dims>(x, cos, sin, out, layout, b, m, h, k, b_stride, m_stride, h_stride, offset,
                                           num_elements);
  }
}

template<typename T>
class FusedApplyRotaryEmbKernel final : public user_op::OpKernel {
 public:
  FusedApplyRotaryEmbKernel() = default;
  ~FusedApplyRotaryEmbKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();

    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* cos = ctx->Tensor4ArgNameAndIndex("cos", 0);
    const user_op::Tensor* sin = ctx->Tensor4ArgNameAndIndex("sin", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const std::string& layout = ctx->Attr<std::string>("layout");

    constexpr size_t N = 4;
    int64_t b = 0;
    int64_t m = 0;
    int64_t h = 0;
    int64_t k = 0;
    int64_t b_stride = 0;
    int64_t m_stride = 0;
    int64_t h_stride = 0;
    int64_t offset   = 0;

    ParseDims(x->shape_view(), layout, Optional<int64_t>(), Optional<int64_t>(cos->shape_view().At(1)), 1,
      &b, &m, &h, &k, &b_stride, &m_stride, &h_stride, &offset);

    // TODO: hard code num_dims & seems redundant template problem...
    DispatchIndex<T, N>(
        reinterpret_cast<const T*>(x->dptr()), reinterpret_cast<const T*>(cos->dptr()),
        reinterpret_cast<const T*>(sin->dptr()), reinterpret_cast<T*>(out->mut_dptr()), layout,
        b, m, h, k, b_stride, m_stride, h_stride, offset);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_APPLY_ROTARY_EMB_GPU(dtype)              \
  REGISTER_USER_KERNEL("fused_apply_rotary_emb")                       \
      .SetCreateFn<FusedApplyRotaryEmbKernel<dtype>>()                \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_APPLY_ROTARY_EMB_GPU(float);
REGISTER_FUSED_APPLY_ROTARY_EMB_GPU(half);
#if CUDA_VERSION >= 11000
REGISTER_FUSED_APPLY_ROTARY_EMB_GPU(nv_bfloat16);
#endif  // CUDA_VERSION >= 11000

}  // namespace

}  // namespace oneflow
