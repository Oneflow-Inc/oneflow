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

void ParseDims(const ShapeView& shape, const std::string& x_layout,
               const Optional<int64_t>& num_heads, const Optional<int64_t>& head_size,
               int64_t tensor_index, int64_t* b, int64_t* m, int64_t* h, int64_t* k,
               int64_t* b_stride, int64_t* m_stride, int64_t* h_stride, int64_t* offset) {
  if (shape.NumAxes() == 3) {
    if (x_layout == "BM(HK)" || x_layout == "BM(H2K)" || x_layout == "BM(H3K)"
        || x_layout == "MB(HK)" || x_layout == "MB(H2K)" || x_layout == "MB(H3K)") {
      bool batch_first = false;
      int64_t packed_n = 0;
      const std::string x_layout_bm = x_layout.substr(0, 2);
      const std::string x_layout_hk = x_layout.substr(2);
      if (x_layout_bm == "BM") {
        *b = shape.At(0);
        *m = shape.At(1);
        batch_first = true;
      } else if (x_layout_bm == "MB") {
        *b = shape.At(1);
        *m = shape.At(0);
        batch_first = false;
      } else {
        UNIMPLEMENTED();
      }
      if (x_layout_hk == "(HK)") {
        packed_n = 1;
      } else if (x_layout_hk == "(H2K)") {
        packed_n = 2;
      } else if (x_layout_hk == "(H3K)") {
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
    if (x_layout == "BMHK") {
      *b = shape.At(0);
      *m = shape.At(1);
      *h = shape.At(2);
      *k = shape.At(3);
      *h_stride = *k;
      *m_stride = *h_stride * *h;
      *b_stride = *m_stride * *m;
    } else if (x_layout == "BHMK") {
      *b = shape.At(0);
      *m = shape.At(2);
      *h = shape.At(1);
      *k = shape.At(3);
      *m_stride = *k;
      *h_stride = *m_stride * *m;
      *b_stride = *h_stride * *h;
    } else if (x_layout == "MBHK") {
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

template<typename T, typename PositionType, typename IndexType, size_t NumDims, size_t RotaryEmbDim>
struct FusedApplyRotaryEmbParam {
  const T* x;
  const T* cos;
  const T* sin;
  const PositionType* position_ids;
  T* out;
  T theta;
  IndexType rotary_size;
  IndexType rotate_stride;
  IndexType k0;
  IndexType k1;
  IndexType num_elements;
  IndexType k;
  IndexType packed_n;
  IndexType offset;
  std::pair<char, IndexType> out_stride[NumDims];  // ordered descendingly by stride

  IndexType x_b_stride;
  IndexType x_m_stride;
  IndexType x_h_stride;

  IndexType position_b_stride;
  IndexType position_rotate_stride;

  IndexType sinuous_m_stride;

  FusedApplyRotaryEmbParam(const T* x, const T* cos, const T* sin, const PositionType* position_ids,
                           T* out, const T theta, const IndexType rotary_size,
                           const IndexType rotate_stride, const IndexType num_elements,
                           const IndexType k, const IndexType k0, const IndexType k1,
                           const IndexType offset, const IndexType packed_n)
      : x(x),
        cos(cos),
        sin(sin),
        position_ids(position_ids),
        out(out),
        theta(theta),
        rotary_size(rotary_size),
        rotate_stride(rotate_stride),
        num_elements(num_elements),
        k(k),
        k0(k0),
        k1(k1),
        offset(offset),
        packed_n(packed_n) {}
};

template<typename T, typename PositionType, typename IndexType, size_t PackSize, size_t NumDims,
         size_t RotaryEmbDim>
__global__ void IntervalKernel(
    FusedApplyRotaryEmbParam<T, PositionType, IndexType, NumDims, RotaryEmbDim> param) {
  for (IndexType packed_offset = threadIdx.x + blockIdx.x * blockDim.x;
       packed_offset < param.num_elements; packed_offset += blockDim.x * gridDim.x) {
    using LoadPack = cuda::elementwise::Packed<T, PackSize>;
    IndexType offset = packed_offset * PackSize;
    IndexType b_index = 0, m_index = 0, h_index = 0, k_index = 0;

    IndexType temp_offset = offset;

    for (int i = 0; i < NumDims; i++) {
      IndexType dim_tag = param.out_stride[i].first;
      IndexType out_stride = param.out_stride[i].second;
      IndexType index = temp_offset / out_stride;
      if (dim_tag == 'b') {
        b_index = index;
      } else if (dim_tag == 'm') {
        m_index = index;
      } else if (dim_tag == 'h') {
        h_index = index;
      } else {
        k_index = index;
      }
      temp_offset = temp_offset - index * out_stride;
    }

    const IndexType x_offset = param.x_b_stride * b_index + param.x_m_stride * m_index
                               + param.x_h_stride * h_index + k_index + param.offset;
    const LoadPack x_vec = *reinterpret_cast<const LoadPack*>(param.x + x_offset);

    if (k_index < param.rotary_size) {
      const IndexType position_rotate_index = (k_index >= param.k0) ? 1 : 0;
      const IndexType position_id_offset = b_index * param.position_b_stride
                                           + position_rotate_index * param.position_rotate_stride
                                           + m_index;

      const PositionType position =
          param.position_ids ? param.position_ids[position_id_offset] : m_index;
      const IndexType sinuous_offset = position * param.sinuous_m_stride + k_index;

      LoadPack cos_vec, sin_vec, out_vec;

      if (param.cos && param.sin) {
        cos_vec = *reinterpret_cast<const LoadPack*>(param.cos + sinuous_offset);
        sin_vec = *reinterpret_cast<const LoadPack*>(param.sin + sinuous_offset);
      } else {
        const IndexType actual_ndim = param.rotary_size / RotaryEmbDim;
#pragma unloop
        for (int i = 0; i < PackSize / 2; i++) {
          float val = position
                      * expf(2.0f * static_cast<float>(((k_index % actual_ndim) / 2 + i))
                             / actual_ndim * logf(param.theta));
          T cos_val = cosf(val);
          T sin_val = sinf(val);
          cos_vec.elem[i * 2] = cos_val;
          cos_vec.elem[i * 2 + 1] = cos_val;
          sin_vec.elem[i * 2] = sin_val;
          sin_vec.elem[i * 2 + 1] = sin_val;
        }
      }

#pragma unloop
      for (int i = 0; i < PackSize / 2; i++) {
        out_vec.elem[i * 2] =
            x_vec.elem[i * 2] * cos_vec.elem[i * 2] - x_vec.elem[i * 2 + 1] * sin_vec.elem[i * 2];
        out_vec.elem[i * 2 + 1] = x_vec.elem[i * 2 + 1] * cos_vec.elem[i * 2 + 1]
                                  + x_vec.elem[i * 2] * sin_vec.elem[i * 2 + 1];
      }

      *(reinterpret_cast<LoadPack*>(param.out + offset)) = out_vec;
    } else {
      *(reinterpret_cast<LoadPack*>(param.out + offset)) = x_vec;
    }
  }
}

template<typename T, typename PositionType, typename IndexType, size_t NumDims, size_t RotaryEmbDim>
__global__ void PlaneKernel(
    FusedApplyRotaryEmbParam<T, PositionType, IndexType, NumDims, RotaryEmbDim> param) {
  for (IndexType offset = threadIdx.x + blockIdx.x * blockDim.x; offset < param.num_elements;
       offset += blockDim.x * gridDim.x) {
    using LoadPack = cuda::elementwise::Packed<T, 2>;
    IndexType temp_offset = offset;
    IndexType b_index = 0, m_index = 0, h_index = 0, k_index = 0;
#pragma unloop
    for (int i = 0; i < NumDims; i++) {
      IndexType dim_tag = param.out_stride[i].first;
      IndexType out_stride = param.out_stride[i].second;
      IndexType index = temp_offset / out_stride;
      if (dim_tag == 'b') {
        b_index = index;
      } else if (dim_tag == 'm') {
        m_index = index;
      } else if (dim_tag == 'h') {
        h_index = index;
      } else {
        k_index = index;
      }
      temp_offset = temp_offset - index * out_stride;
    }

    const IndexType position_rotate_index = (k_index >= param.k0) ? 1 : 0;
    const IndexType position_id_offset = b_index * param.position_b_stride
                                         + position_rotate_index * param.position_rotate_stride
                                         + m_index;

    const PositionType position =
        param.position_ids ? param.position_ids[position_id_offset] : m_index;
    const IndexType sinuous_offset = position * param.k + k_index;

    T cos_val, sin_val, out_val;

    if (param.cos && param.sin) {
      cos_val = *(param.cos + sinuous_offset);
      sin_val = *(param.sin + sinuous_offset);
    } else {
      int actual_ndim = param.rotary_size / RotaryEmbDim;
      float val = position
                  * expf(2.0f * static_cast<float>(k_index % (actual_ndim / 2)) / actual_ndim
                         * logf(param.theta));
      cos_val = cosf(val);
      sin_val = sinf(val);
    }

    LoadPack x_vec;
    const IndexType x_offset = param.x_b_stride * b_index + param.x_m_stride * m_index
                               + param.x_h_stride * h_index + k_index + param.offset;

    if (k_index < param.k0) {
      x_vec.elem[0] = *(param.x + x_offset);
      x_vec.elem[1] = (param.k0 - k_index > param.rotate_stride)
                          ? static_cast<T>(-*(param.x + x_offset + param.rotate_stride))
                          : *(param.x + x_offset - param.rotate_stride);
      out_val = cos_val * x_vec.elem[0] + sin_val * x_vec.elem[1];
    } else if (k_index < param.k1) {
      x_vec.elem[0] = *(param.x + x_offset);
      x_vec.elem[1] = (param.k1 - k_index > param.rotate_stride)
                          ? static_cast<T>(-*(param.x + x_offset + param.rotate_stride))
                          : *(param.x + x_offset - param.rotate_stride);
      out_val = cos_val * x_vec.elem[0] + sin_val * x_vec.elem[1];
    } else {
      out_val = *(param.x + x_offset);
    }

    *(param.out + offset) = out_val;
  }
}

template<typename T, typename PositionType, typename IndexType, size_t PackSize, size_t NumDims,
         size_t RotaryEmbDim>
void LaunchKernel(ep::CudaStream* stream, const T* x, const T* cos, const T* sin,
                  const PositionType* position_ids, T* out, const int64_t* position_shape,
                  const std::string& x_layout, const std::string& output_layout,
                  const std::string& mode, const T theta, const int64_t rotary_size,
                  const int64_t b, const int64_t m, const int64_t h, const int64_t k,
                  const int64_t b_stride, const int64_t m_stride, const int64_t h_stride,
                  const int64_t offset, IndexType num_elements) {
  DimVector kernel_x_shape(NumDims), kernel_sinuous_shape(NumDims);

  const IndexType k0 = rotary_size / RotaryEmbDim,
                  k1 = rotary_size;  // TODO: this only support 1d, 2d, rotary postional encoding

  const IndexType rotate_stride = rotary_size / (2 * RotaryEmbDim);

  IndexType packed_n = 1;
  // TODO: only to test its accuracy, needs to be passed through parseDims
  if (x_layout == "BM(H2K)" || x_layout == "MB(H2K)") {
    packed_n = 2;
  } else if (x_layout == "BM(H3K)" || x_layout == "MB(H3K)") {
    packed_n = 3;
  }

  struct FusedApplyRotaryEmbParam<T, PositionType, IndexType, NumDims, RotaryEmbDim> param(
      x, cos, sin, position_ids, out, theta, rotary_size, rotate_stride, num_elements, k, k0, k1,
      offset, packed_n);

  std::pair<char, IndexType> strides[NumDims];
  strides[0] = {'b', b_stride / packed_n};
  strides[1] = {'h', h_stride / packed_n};
  strides[2] = {'m', m_stride / packed_n};
  strides[3] = {'k', 1};

  param.x_b_stride = b_stride;
  param.x_h_stride = h_stride;
  param.x_m_stride = m_stride;

  param.sinuous_m_stride = k;

  const IndexType position_m = position_shape ? static_cast<IndexType>(position_shape[2]) : m;
  param.position_rotate_stride = position_m;
  param.position_b_stride = position_m * RotaryEmbDim;

  auto GetOutDim = [&](const char c) {
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

  std::sort(strides, strides + NumDims, [&](auto pair1, auto pair2) {
    if (pair1.second > pair2.second) {
      return true;
    } else if (pair1.second == pair2.second) {
      if (GetOutDim(pair1.first) != 1) { return true; }
      return false;
    } else {
      return false;
    }
    return pair1.second > pair2.second;
  });

// K has to be the last dimension, only k&m matters, therefore strides other than k&m does not
// really needs to be computed
#pragma unloop
  for (int i = 0; i < NumDims; i++) { param.out_stride[i] = strides[i]; }

  constexpr size_t blk_size = 128;

  if (mode == "plane") {
    param.num_elements = param.num_elements * PackSize;
    PlaneKernel<T, PositionType, IndexType, NumDims, RotaryEmbDim>
        <<<(param.num_elements + blk_size - 1) / blk_size, blk_size, 0, stream->cuda_stream()>>>(
            param);
  } else {
    IntervalKernel<T, PositionType, IndexType, PackSize, NumDims, RotaryEmbDim>
        <<<(param.num_elements + blk_size - 1) / blk_size, blk_size, 0, stream->cuda_stream()>>>(
            param);
  }
}

template<typename T, typename PositionType, typename IndexType, size_t NumDims, size_t RotaryEmbDim>
void DispatchPackSize(ep::CudaStream* stream, const T* x, const T* cos, const T* sin,
                      const PositionType* position_ids, T* out, const int64_t* position_shape,
                      const std::string& x_layout, const std::string& output_layout,
                      const std::string& mode, const T theta, const int64_t rotary_size,
                      const IndexType b, const IndexType m, const IndexType h, const IndexType k,
                      const IndexType b_stride, const IndexType m_stride, const IndexType h_stride,
                      const IndexType offset, IndexType num_elements) {
  const auto CheckPackSize = [&](const size_t PackSize) {
    bool r = (((reinterpret_cast<uintptr_t>(x) % (sizeof(T) * PackSize)) == 0)
              && (((rotary_size / RotaryEmbDim) % PackSize) == 0)
              && (((k - rotary_size) % PackSize) == 0) && ((16 / sizeof(T)) >= PackSize));
    return r;
  };

  if (CheckPackSize(8)) {
    num_elements /= 8;
    LaunchKernel<T, PositionType, IndexType, 8, NumDims, RotaryEmbDim>(
        stream, x, cos, sin, position_ids, out, position_shape, x_layout, output_layout, mode,
        theta, rotary_size, b, m, h, k, b_stride, m_stride, h_stride, offset, num_elements);
  } else if (CheckPackSize(4)) {
    num_elements /= 4;
    LaunchKernel<T, PositionType, IndexType, 4, NumDims, RotaryEmbDim>(
        stream, x, cos, sin, position_ids, out, position_shape, x_layout, output_layout, mode,
        theta, rotary_size, b, m, h, k, b_stride, m_stride, h_stride, offset, num_elements);
  } else {
    num_elements /= 2;
    LaunchKernel<T, PositionType, IndexType, 2, NumDims, RotaryEmbDim>(
        stream, x, cos, sin, position_ids, out, position_shape, x_layout, output_layout, mode,
        theta, rotary_size, b, m, h, k, b_stride, m_stride, h_stride, offset, num_elements);
  }
}

template<typename T, typename PositionType, size_t NumDims, size_t RotaryEmbDim>
void DispatchIndex(ep::CudaStream* stream, const T* x, const T* cos, const T* sin,
                   const PositionType* position_ids, T* out, const int64_t* position_shape,
                   const std::string& x_layout, const std::string& output_layout,
                   const std::string& mode, const T theta, const int64_t rotary_size,
                   const int64_t b, const int64_t m, const int64_t h, const int64_t k,
                   const int64_t b_stride, const int64_t m_stride, const int64_t h_stride,
                   const int64_t offset) {
  int64_t num_elements = b * m * h * k;
  if (num_elements < (1 << 30)) {
    DispatchPackSize<T, PositionType, int32_t, NumDims, RotaryEmbDim>(
        stream, x, cos, sin, position_ids, out, position_shape, x_layout, output_layout, mode,
        theta, rotary_size, static_cast<int32_t>(b), static_cast<int32_t>(m),
        static_cast<int32_t>(h), static_cast<int32_t>(k), static_cast<int32_t>(b_stride),
        static_cast<int32_t>(m_stride), static_cast<int32_t>(h_stride),
        static_cast<int32_t>(offset), static_cast<int32_t>(num_elements));
  } else {
    DispatchPackSize<T, PositionType, int64_t, NumDims, RotaryEmbDim>(
        stream, x, cos, sin, position_ids, out, position_shape, x_layout, output_layout, mode,
        theta, rotary_size, b, m, h, k, b_stride, m_stride, h_stride, offset, num_elements);
  }
}

template<typename T, typename PositionType, size_t NumDims>
void DispatchRotaryEmbeddingDimension(ep::CudaStream* stream, const T* x, const T* cos,
                                      const T* sin, const PositionType* position_ids, T* out,
                                      const int64_t* position_shape, const std::string& x_layout,
                                      const std::string& output_layout, const std::string& mode,
                                      const T theta, const int64_t rotary_size,
                                      const int rotary_emb_dim, const int64_t b, const int64_t m,
                                      const int64_t h, const int64_t k, const int64_t b_stride,
                                      const int64_t m_stride, const int64_t h_stride,
                                      const int64_t offset) {
  if (rotary_emb_dim == 1) {
    DispatchIndex<T, PositionType, NumDims, 1>(
        stream, x, cos, sin, position_ids, out, position_shape, x_layout, output_layout, mode,
        theta, rotary_size, b, m, h, k, b_stride, m_stride, h_stride, offset);
  } else if (rotary_emb_dim == 2) {
    DispatchIndex<T, PositionType, NumDims, 2>(
        stream, x, cos, sin, position_ids, out, position_shape, x_layout, output_layout, mode,
        theta, rotary_size, b, m, h, k, b_stride, m_stride, h_stride, offset);
  }
}

template<typename T, typename PositionType>
class FusedApplyRotaryEmbKernel final : public user_op::OpKernel {
 public:
  FusedApplyRotaryEmbKernel() = default;
  ~FusedApplyRotaryEmbKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* cos = nullptr;
    user_op::Tensor* sin = nullptr;
    user_op::Tensor* position_ids = nullptr;
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const std::string& x_layout = ctx->Attr<std::string>("x_layout");
    const std::string& output_layout = ctx->Attr<std::string>("output_layout");
    const std::string& mode = ctx->Attr<std::string>("mode");
    const int64_t tensor_index = ctx->Attr<int64_t>("tensor_index");
    const int64_t k_size = ctx->Attr<int64_t>("k_size");
    const int64_t rotary_size = ctx->Attr<int64_t>("rotary_size");
    const float theta = 1.0f / ctx->Attr<float>("base");
    int rotary_emb_dim = 1;

    if (ctx->has_input("cos", 0)) { cos = ctx->Tensor4ArgNameAndIndex("cos", 0); }

    if (ctx->has_input("sin", 0)) { sin = ctx->Tensor4ArgNameAndIndex("sin", 0); }

    if (ctx->has_input("position_ids", 0)) {
      position_ids = ctx->Tensor4ArgNameAndIndex("position_ids", 0);
      rotary_emb_dim = position_ids->shape_view().At(1);
    }

    constexpr size_t ndims = 4;
    int64_t b = 0;
    int64_t m = 0;
    int64_t h = 0;
    int64_t k = 0;
    int64_t b_stride = 0;
    int64_t m_stride = 0;
    int64_t h_stride = 0;
    int64_t offset = 0;

    ParseDims(x->shape_view(), x_layout, Optional<int64_t>(),
              k_size ? Optional<int64_t>(k_size) : Optional<int64_t>(), tensor_index, &b, &m, &h,
              &k, &b_stride, &m_stride, &h_stride, &offset);

    // TODO: hard code NumDims & seems redundant template problem...
    DispatchRotaryEmbeddingDimension<T, PositionType, ndims>(
        ctx->stream()->As<ep::CudaStream>(), reinterpret_cast<const T*>(x->dptr()),
        cos ? reinterpret_cast<const T*>(cos->dptr()) : nullptr,
        sin ? reinterpret_cast<const T*>(sin->dptr()) : nullptr,
        position_ids ? reinterpret_cast<const PositionType*>(position_ids->dptr()) : nullptr,
        reinterpret_cast<T*>(out->mut_dptr()),
        position_ids ? position_ids->shape_view().data() : nullptr, x_layout, output_layout, mode,
        static_cast<T>(theta), rotary_size, rotary_emb_dim, b, m, h, k, b_stride, m_stride,
        h_stride, offset);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_APPLY_ROTARY_EMB_GPU(dtype, position_type)          \
  REGISTER_USER_KERNEL("fused_apply_rotary_emb")                           \
      .SetCreateFn<FusedApplyRotaryEmbKernel<dtype, position_type>>()      \
      .SetIsMatchedHob(                                                    \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                  \
          && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value) \
          && (user_op::HobInputSize("position_ids") == 1)                  \
          && (user_op::HobDataType("position_ids", 0) == GetDataType<position_type>::value));

#define REGISTER_FUSED_APPLY_ROTARY_EMB_GPU_DTYPE(dtype)                                \
  REGISTER_FUSED_APPLY_ROTARY_EMB_GPU(dtype, int64_t);                                  \
  REGISTER_FUSED_APPLY_ROTARY_EMB_GPU(dtype, int32_t);                                  \
  REGISTER_USER_KERNEL("fused_apply_rotary_emb")                                        \
      .SetCreateFn<FusedApplyRotaryEmbKernel<dtype, int64_t>>()                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobInputSize("position_ids") == 0));

REGISTER_FUSED_APPLY_ROTARY_EMB_GPU_DTYPE(float);
REGISTER_FUSED_APPLY_ROTARY_EMB_GPU_DTYPE(half);
#if CUDA_VERSION >= 11000
REGISTER_FUSED_APPLY_ROTARY_EMB_GPU_DTYPE(nv_bfloat16);
#endif  // CUDA_VERSION >= 11000

}  // namespace

}  // namespace oneflow
