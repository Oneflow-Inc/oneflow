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

template<typename T, typename PositionType, typename IndexType, int num_dims>
struct FusedApplyRotaryEmbParam {
  const T* x;
  const T* cos;
  const T* sin;
  const PositionType* position_ids;
  T* out;
  T theta;
  int64_t pass_ndims;
  int64_t rotary_emb_dim;  // TODO: dispatch
  IndexType k0;
  IndexType k1;
  IndexType num_elements;
  IndexType k;
  IndexType sinuous_m;
  IndexType offset;
  IndexType x_stride[num_dims];
  IndexType position_stride[num_dims];
  IndexType position_mask[num_dims];
  IndexType sinuous_stride[num_dims];
  IndexType sinuous_mask[num_dims];

  // TODO: position_id type? for now it is int32
  FusedApplyRotaryEmbParam(const T* x, const T* cos, const T* sin, const PositionType* position_ids,
                           T* out, const T theta, const int64_t pass_ndims,
                           const int rotary_emb_dim, const IndexType num_elements,
                           const IndexType k, const IndexType k0, const IndexType k1,
                           const IndexType sinuous_m, const IndexType offset)
      : x(x),
        cos(cos),
        sin(sin),
        position_ids(position_ids),
        out(out),
        theta(theta),
        pass_ndims(pass_ndims),
        rotary_emb_dim(rotary_emb_dim),
        num_elements(num_elements),
        k(k),
        k0(k0),
        k1(k1),
        sinuous_m(sinuous_m),
        offset(offset) {}
};

template<typename T, typename PositionType, typename IndexType, size_t pack_size, size_t num_dims>
__global__ void FusedApplyRotaryEmbFetchKernel(
    FusedApplyRotaryEmbParam<T, PositionType, IndexType, num_dims> param) {
  const T* x = param.x;
  const T* cos = param.cos;
  const T* sin = param.sin;
  const PositionType* position_ids = param.position_ids;
  T* out = param.out;
  const IndexType packed_k = param.k / pack_size;
  const IndexType packed_pass_ndims = param.pass_ndims / pack_size;
  const IndexType packed_rotary_ndims = packed_k - packed_pass_ndims;
  const IndexType rotary_num_elements = param.num_elements / packed_k * packed_rotary_ndims;
  for (IndexType packed_offset = threadIdx.x + blockIdx.x * blockDim.x;
       packed_offset < rotary_num_elements; packed_offset += blockDim.x * gridDim.x) {
    using LoadPack = cuda::elementwise::Packed<T, pack_size>;
    IndexType offset =
        param.offset
        + (packed_offset % packed_rotary_ndims + packed_offset / packed_rotary_ndims * packed_k)
              * pack_size;
    const LoadPack* x_load = reinterpret_cast<const LoadPack*>(x + offset);
    const LoadPack x_vec = *x_load;
    IndexType sinuous_offset = 0;
    IndexType position_id_offset = 0;
    IndexType k_index = 0;

    IndexType temp_offset = offset;
#pragma unloop
    for (int i = 0; i < num_dims; i++) {
      IndexType index = temp_offset / param.x_stride[i];
      if (i == num_dims - 1) k_index = index;
      temp_offset = temp_offset - index * param.x_stride[i];
      position_id_offset =
          position_id_offset + (index * param.position_mask[i] * param.position_stride[i]);
    }

    position_id_offset =
        position_id_offset
        + param.sinuous_m * (k_index * param.rotary_emb_dim / (param.k - param.pass_ndims));

    const int position = position_ids[position_id_offset];

    sinuous_offset = position * param.k + k_index;

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

    *(reinterpret_cast<LoadPack*>(out + offset)) = out_vec;
  }

  for (IndexType packed_offset = threadIdx.x + blockIdx.x * blockDim.x;
       packed_offset < param.num_elements && packed_offset >= rotary_num_elements;
       packed_offset += blockDim.x * gridDim.x) {
    using LoadPack = cuda::elementwise::Packed<T, pack_size>;
    IndexType offset =
        param.offset
        + (packed_rotary_ndims + (packed_offset - rotary_num_elements) % packed_pass_ndims
           + (packed_offset - rotary_num_elements) / packed_pass_ndims * packed_k)
              * pack_size;
    const LoadPack* x_load = reinterpret_cast<const LoadPack*>(x + offset);
    const LoadPack x_vec = *x_load;

    *(reinterpret_cast<LoadPack*>(out + offset)) = x_vec;
  }
}

template<typename T, typename PositionType, typename IndexType, size_t pack_size, size_t num_dims>
__global__ void FusedApplyRotaryEmbFetchWithoutPositionKernel(
    FusedApplyRotaryEmbParam<T, PositionType, IndexType, num_dims> param) {
  const T* x = param.x;
  const T* cos = param.cos;
  const T* sin = param.sin;
  T* out = param.out;
  const IndexType packed_k = param.k / pack_size;
  const IndexType packed_pass_ndims = param.pass_ndims / pack_size;
  const IndexType packed_rotary_ndims = packed_k - packed_pass_ndims;
  const IndexType rotary_num_elements = param.num_elements / packed_k * packed_rotary_ndims;
  for (IndexType packed_offset = threadIdx.x + blockIdx.x * blockDim.x;
       packed_offset < rotary_num_elements; packed_offset += blockDim.x * gridDim.x) {
    using LoadPack = cuda::elementwise::Packed<T, pack_size>;
    IndexType offset =
        param.offset
        + (packed_offset % packed_rotary_ndims + packed_offset / packed_rotary_ndims * packed_k)
              * pack_size;
    const LoadPack* x_load = reinterpret_cast<const LoadPack*>(x + offset);
    const LoadPack x_vec = *x_load;
    IndexType sinuous_offset = 0;
    IndexType k_index = 0;

    IndexType temp_offset = offset;
#pragma unloop
    for (int i = 0; i < num_dims; i++) {
      IndexType index = temp_offset / param.x_stride[i];
      temp_offset = temp_offset - index * param.x_stride[i];
      sinuous_offset = sinuous_offset + index * param.sinuous_stride[i] * param.sinuous_mask[i];
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

    *(reinterpret_cast<LoadPack*>(out + offset)) = out_vec;
  }

  for (IndexType packed_offset = threadIdx.x + blockIdx.x * blockDim.x;
       packed_offset < param.num_elements && packed_offset >= rotary_num_elements;
       packed_offset += blockDim.x * gridDim.x) {
    using LoadPack = cuda::elementwise::Packed<T, pack_size>;
    IndexType offset =
        param.offset
        + (packed_rotary_ndims + (packed_offset - rotary_num_elements) % packed_pass_ndims
           + (packed_offset - rotary_num_elements) / packed_pass_ndims * packed_k)
              * pack_size;
    const LoadPack* x_load = reinterpret_cast<const LoadPack*>(x + offset);
    const LoadPack x_vec = *x_load;

    *(reinterpret_cast<LoadPack*>(out + offset)) = x_vec;
  }
}

template<typename T, typename PositionType, typename IndexType, size_t pack_size, size_t num_dims>
__global__ void FusedApplyRotaryEmbComputeWithoutPositionKernel(
    FusedApplyRotaryEmbParam<T, PositionType, IndexType, num_dims> param) {
  const T* x = param.x;
  const T* cos = param.cos;
  const T* sin = param.sin;
  T* out = param.out;
  const T theta = param.theta;
  const IndexType packed_k = param.k / pack_size;
  const IndexType packed_pass_ndims = param.pass_ndims / pack_size;
  const IndexType packed_rotary_ndims = packed_k - packed_pass_ndims;
  const IndexType rotary_num_elements = param.num_elements / packed_k * packed_rotary_ndims;
  for (IndexType packed_offset = threadIdx.x + blockIdx.x * blockDim.x;
       packed_offset < rotary_num_elements; packed_offset += blockDim.x * gridDim.x) {
    using LoadPack = cuda::elementwise::Packed<T, pack_size>;
    IndexType offset =
        param.offset
        + (packed_offset % packed_rotary_ndims + packed_offset / packed_rotary_ndims * packed_k)
              * pack_size;
    const LoadPack* x_load = reinterpret_cast<const LoadPack*>(x + offset);
    const LoadPack x_vec = *x_load;
    IndexType sinuous_offset = 0;
    IndexType k_index = 0;

    IndexType temp_offset = offset;
#pragma unloop
    for (int i = 0; i < num_dims; i++) {
      IndexType index = temp_offset / param.x_stride[i];
      if (i == num_dims - 1) k_index = index;
      temp_offset = temp_offset - index * param.x_stride[i];
      sinuous_offset = sinuous_offset + index * param.sinuous_stride[i] * param.sinuous_mask[i];
    }

    IndexType m_index = sinuous_offset / param.k;

    LoadPack cos_vec, sin_vec, out_vec;

#pragma unloop
    for (int i = 0; i < pack_size / 2; i++) {
      float val =
          m_index * expf(2.0f * static_cast<float>(k_index / 2 + i) / param.k * logf(theta));
      T cos_val = cosf(val);
      T sin_val = sinf(val);
      cos_vec.elem[i * 2] = cos_val;
      cos_vec.elem[i * 2 + 1] = cos_val;
      sin_vec.elem[i * 2] = sin_val;
      sin_vec.elem[i * 2 + 1] = sin_val;
    }

#pragma unloop
    for (int i = 0; i < pack_size / 2; i++) {
      out_vec.elem[i * 2] =
          x_vec.elem[i * 2] * cos_vec.elem[i * 2] - x_vec.elem[i * 2 + 1] * sin_vec.elem[i * 2];
      out_vec.elem[i * 2 + 1] = x_vec.elem[i * 2 + 1] * cos_vec.elem[i * 2 + 1]
                                + x_vec.elem[i * 2] * sin_vec.elem[i * 2 + 1];
    }

    *(reinterpret_cast<LoadPack*>(out + offset)) = out_vec;
  }

  for (IndexType packed_offset = threadIdx.x + blockIdx.x * blockDim.x;
       packed_offset < param.num_elements && packed_offset >= rotary_num_elements;
       packed_offset += blockDim.x * gridDim.x) {
    using LoadPack = cuda::elementwise::Packed<T, pack_size>;
    IndexType offset =
        param.offset
        + (packed_rotary_ndims + (packed_offset - rotary_num_elements) % packed_pass_ndims
           + (packed_offset - rotary_num_elements) / packed_pass_ndims * packed_k)
              * pack_size;
    const LoadPack* x_load = reinterpret_cast<const LoadPack*>(x + offset);
    const LoadPack x_vec = *x_load;

    *(reinterpret_cast<LoadPack*>(out + offset)) = x_vec;
  }
}

template<typename T, typename PositionType, typename IndexType, size_t pack_size, size_t num_dims>
__global__ void FusedApplyRotaryEmbComputeKernel(
    FusedApplyRotaryEmbParam<T, PositionType, IndexType, num_dims> param) {
  const T* x = param.x;
  const T* cos = param.cos;
  const T* sin = param.sin;
  const PositionType* position_ids = param.position_ids;
  T* out = param.out;
  const T theta = param.theta;
  const IndexType packed_k = param.k / pack_size;
  const IndexType packed_pass_ndims = param.pass_ndims / pack_size;
  const IndexType packed_rotary_ndims = packed_k - packed_pass_ndims;
  const IndexType rotary_num_elements = param.num_elements / packed_k * packed_rotary_ndims;
  for (IndexType packed_offset = threadIdx.x + blockIdx.x * blockDim.x;
       packed_offset < rotary_num_elements; packed_offset += blockDim.x * gridDim.x) {
    using LoadPack = cuda::elementwise::Packed<T, pack_size>;
    IndexType offset =
        param.offset
        + (packed_offset % packed_rotary_ndims + packed_offset / packed_rotary_ndims * packed_k)
              * pack_size;
    const LoadPack* x_load = reinterpret_cast<const LoadPack*>(x + offset);
    const LoadPack x_vec = *x_load;
    IndexType position_id_offset = 0;
    IndexType k_index = 0;

    IndexType temp_offset = offset;
#pragma unloop
    for (int i = 0; i < num_dims; i++) {
      IndexType index = temp_offset / param.x_stride[i];
      if (i == num_dims - 1) k_index = index;
      temp_offset = temp_offset - index * param.x_stride[i];
      position_id_offset =
          position_id_offset + (index * param.position_mask[i] * param.position_stride[i]);
    }

    position_id_offset =
        position_id_offset
        + param.sinuous_m * (k_index * param.rotary_emb_dim / (param.k - param.pass_ndims));

    const PositionType position = position_ids[position_id_offset];

    LoadPack cos_vec, sin_vec, out_vec;

#pragma unloop
    for (int i = 0; i < pack_size / 2; i++) {
      float val =
          position * expf(2.0f * static_cast<float>(k_index / 2 + i) / param.k * logf(theta));
      T cos_val = cosf(val);
      T sin_val = sinf(val);
      cos_vec.elem[i * 2] = cos_val;
      cos_vec.elem[i * 2 + 1] = cos_val;
      sin_vec.elem[i * 2] = sin_val;
      sin_vec.elem[i * 2 + 1] = sin_val;
    }

#pragma unloop
    for (int i = 0; i < pack_size / 2; i++) {
      out_vec.elem[i * 2] =
          x_vec.elem[i * 2] * cos_vec.elem[i * 2] - x_vec.elem[i * 2 + 1] * sin_vec.elem[i * 2];
      out_vec.elem[i * 2 + 1] = x_vec.elem[i * 2 + 1] * cos_vec.elem[i * 2 + 1]
                                + x_vec.elem[i * 2] * sin_vec.elem[i * 2 + 1];
    }

    *(reinterpret_cast<LoadPack*>(out + offset)) = out_vec;
  }

  for (IndexType packed_offset = threadIdx.x + blockIdx.x * blockDim.x;
       packed_offset < param.num_elements && packed_offset >= rotary_num_elements;
       packed_offset += blockDim.x * gridDim.x) {
    using LoadPack = cuda::elementwise::Packed<T, pack_size>;
    IndexType offset =
        param.offset
        + (packed_rotary_ndims + (packed_offset - rotary_num_elements) % packed_pass_ndims
           + (packed_offset - rotary_num_elements) / packed_pass_ndims * packed_k)
              * pack_size;
    const LoadPack* x_load = reinterpret_cast<const LoadPack*>(x + offset);
    const LoadPack x_vec = *x_load;

    *(reinterpret_cast<LoadPack*>(out + offset)) = x_vec;
  }
}

template<typename T, typename PositionType, typename IndexType, size_t num_dims>
__global__ void PlaneKernel(FusedApplyRotaryEmbParam<T, PositionType, IndexType, num_dims> param) {
  const T* x = param.x;
  const T* cos = param.cos;
  const T* sin = param.sin;
  const PositionType* position_ids = param.position_ids;
  T* out = param.out;
  IndexType sinuous_offset = 0;
  const T theta = param.theta;

  for (IndexType offset = threadIdx.x + blockIdx.x * blockDim.x; offset < param.num_elements;
       offset += blockDim.x * gridDim.x) {
    using LoadPack = cuda::elementwise::Packed<T, 2>;
    IndexType temp_offset = offset;
    IndexType k_index, m_index;
    IndexType position_id_offset = 0;
    PositionType position;
#pragma unloop
    for (int i = 0; i < num_dims; i++) {
      IndexType index = temp_offset / param.x_stride[i];
      if (i == num_dims - 1) k_index = index;
      temp_offset = temp_offset - index * param.x_stride[i];
      sinuous_offset = sinuous_offset + index * param.sinuous_stride[i] * param.sinuous_mask[i];
      position_id_offset =
          position_id_offset + (index * param.position_mask[i] * param.position_stride[i]);
    }

    m_index = sinuous_offset / param.k;
    position_id_offset =
        position_id_offset
        + param.sinuous_m * (k_index * param.rotary_emb_dim / (param.k - param.pass_ndims));

    if (param.position_ids) {
      position = position_ids[position_id_offset];
      sinuous_offset = position * param.k + k_index;
    } else {
      position = m_index;
    }

    LoadPack x_vec;

    T cos_val;
    T sin_val;
    T out_val;

    if (param.cos) {
      cos_val = *(cos + sinuous_offset);
      sin_val = *(sin + sinuous_offset);
    } else {
      int actual_ndim = (param.k - param.pass_ndims) / param.rotary_emb_dim;
      float val = position
                  * expf(2.0f * static_cast<float>(k_index % (actual_ndim / 2)) / actual_ndim
                         * logf(theta));
      cos_val = cosf(val);
      sin_val = sinf(val);
    }

    if (k_index < param.k0) {
      x_vec.elem[0] = *(x + offset);
      x_vec.elem[1] = (k_index < param.k0 / 2) ? static_cast<T>(-*(x + offset + param.k0 / 2))
                                               : *(x + offset - param.k0 / 2);
      out_val = cos_val * x_vec.elem[0] + sin_val * x_vec.elem[1];
    } else if (k_index < param.k1) {
      if ((k_index - param.k0) < ((param.k - param.pass_ndims) / (2 * param.rotary_emb_dim))) {
        x_vec.elem[0] = *(x + offset);
        x_vec.elem[1] = -*(x + offset + param.k0 / 2);
        out_val = cos_val * x_vec.elem[0] + sin_val * x_vec.elem[1];
      } else {
        x_vec.elem[0] = *(x + offset);
        x_vec.elem[1] = *(x + offset - param.k0 / 2);
        out_val = cos_val * x_vec.elem[0] + sin_val * x_vec.elem[1];
      }
    } else {
      *(out + offset) = *(x + offset);
    }

    *(out + offset) = out_val;
  }
}

template<typename T, typename PositionType, typename IndexType, size_t pack_size, size_t num_dims>
void LaunchKernel(ep::CudaStream* stream, const T* x, const T* cos, const T* sin, const PositionType* position_ids, T* out,
                  const int64_t* position_shape, const std::string& x_layout,
                  const std::string& output_layout, const std::string& mode, const T theta,
                  const int64_t pass_ndims, const int rotary_emb_dim, const int64_t b,
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

  IndexType k0, k1;

  if (rotary_emb_dim == 1) {
    k0 = k - pass_ndims;
    k1 = k - pass_ndims;
  } else {
    k0 = (k - pass_ndims) / 2;
    k1 = k - pass_ndims;
  }

  struct FusedApplyRotaryEmbParam<T, PositionType, IndexType, num_dims> param(
      x, cos, sin, position_ids, out, theta, pass_ndims, rotary_emb_dim, num_elements, k, k0, k1,
      position_shape ? static_cast<IndexType>(position_shape[2]) : m, offset);

  std::pair<char, std::int64_t> strides[num_dims];
  strides[0] = {'b', b_stride};
  strides[1] = {'h', h_stride};
  strides[2] = {'m', m_stride};
  strides[3] = {'k', 1};

  auto GetDimX = [&](const char c) {
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
      if (GetDimX(pair1.first) != 1) { return true; }
      return false;
    } else {
      return false;
    }
    return pair1.second > pair2.second;
  });

// K has to be the last dimension, only k&m matters, therefore strides other than k&m does not
// really needs to be computed
#pragma unloop
  for (int i = 0; i < num_dims; i++) {
    param.x_stride[i] = strides[i].second;
    param.sinuous_mask[i] = 0;
    param.position_mask[i] = 0;
    if (strides[i].first == 'm') {
      param.sinuous_mask[i] = 1;
      param.sinuous_stride[i] = k;
      param.position_stride[i] = 1;
      param.position_mask[i] = 1;
    } else if (strides[i].first == 'k') {
      param.sinuous_mask[i] = 1;
      param.sinuous_stride[i] = 1;
    } else if (strides[i].first == 'b') {
      param.position_stride[i] = rotary_emb_dim * param.sinuous_m;
      param.position_mask[i] = 1;
    }
  }

  constexpr size_t blk_size = 128;

  // TODO: something bad here... need to be refined to support mode dispatch
  if (mode == "plane") {
    param.num_elements = param.num_elements * pack_size;
    PlaneKernel<T, PositionType, IndexType, num_dims>
        <<<(param.num_elements + blk_size - 1) / blk_size, blk_size, 0, stream->cuda_stream()>>>(param);
    return;
  }

  if (cos) {
    if (position_ids) {
      FusedApplyRotaryEmbFetchKernel<T, PositionType, IndexType, pack_size, num_dims>
          <<<(param.num_elements + blk_size - 1) / blk_size, blk_size, 0, stream->cuda_stream()>>>(param);
    } else {
      FusedApplyRotaryEmbFetchWithoutPositionKernel<T, PositionType, IndexType, pack_size, num_dims>
          <<<(param.num_elements + blk_size - 1) / blk_size, blk_size, 0, stream->cuda_stream()>>>(param);
    }
  } else {
    if (position_ids) {
      FusedApplyRotaryEmbComputeKernel<T, PositionType, IndexType, pack_size, num_dims>
          <<<(param.num_elements + blk_size - 1) / blk_size, blk_size, 0, stream->cuda_stream()>>>(param);
    } else {
      FusedApplyRotaryEmbComputeWithoutPositionKernel<T, PositionType, IndexType, pack_size,
                                                      num_dims>
          <<<(param.num_elements + blk_size - 1) / blk_size, blk_size, 0, stream->cuda_stream()>>>(param);
    }
  }
}

template<typename T, typename PositionType, typename IndexType, size_t num_dims>
void DispatchPackSize(ep::CudaStream* stream, const T* x, const T* cos, const T* sin, const PositionType* position_ids,
                      T* out, const int64_t* position_shape, const std::string& x_layout,
                      const std::string& output_layout, const std::string& mode, const T theta,
                      const int64_t pass_ndims, const int rotary_emb_dim, const IndexType b,
                      const IndexType m, const IndexType h, const IndexType k,
                      const IndexType b_stride, const IndexType m_stride, const IndexType h_stride,
                      const IndexType offset, IndexType num_elements) {
  const auto CheckPackSize = [&](const size_t pack_size) {
    bool r = (((reinterpret_cast<uintptr_t>(x) % (sizeof(T) * pack_size)) == 0)
              && ((((k - pass_ndims) / 2) % pack_size) == 0) && ((pass_ndims % pack_size) == 0)
              && ((16 / sizeof(T)) >= pack_size));
    return r;
  };

  if (CheckPackSize(8)) {
    num_elements /= 8;
    LaunchKernel<T, PositionType, IndexType, 8, num_dims>(
        stream, x, cos, sin, position_ids, out, position_shape, x_layout, output_layout, mode, theta,
        pass_ndims, rotary_emb_dim, b, m, h, k, b_stride, m_stride, h_stride, offset, num_elements);
  } else if (CheckPackSize(4)) {
    num_elements /= 4;
    LaunchKernel<T, PositionType, IndexType, 4, num_dims>(
        stream, x, cos, sin, position_ids, out, position_shape, x_layout, output_layout, mode, theta,
        pass_ndims, rotary_emb_dim, b, m, h, k, b_stride, m_stride, h_stride, offset, num_elements);
  } else {
    num_elements /= 2;
    LaunchKernel<T, PositionType, IndexType, 2, num_dims>(
        stream, x, cos, sin, position_ids, out, position_shape, x_layout, output_layout, mode, theta,
        pass_ndims, rotary_emb_dim, b, m, h, k, b_stride, m_stride, h_stride, offset, num_elements);
  }
}

template<typename T, typename PositionType, size_t num_dims>
void DispatchIndex(ep::CudaStream* stream, const T* x, const T* cos, const T* sin, const PositionType* position_ids, T* out,
                   const int64_t* position_shape, const std::string& x_layout,
                   const std::string& output_layout, const std::string& mode, const T theta,
                   const int64_t pass_ndims, const int rotary_emb_dim, const int64_t b,
                   const int64_t m, const int64_t h, const int64_t k, const int64_t b_stride,
                   const int64_t m_stride, const int64_t h_stride, const int64_t offset) {
  int64_t num_elements = b * m * h * k;
  if (num_elements < (1 << 30)) {
    DispatchPackSize<T, PositionType, int32_t, num_dims>(
        stream, x, cos, sin, position_ids, out, position_shape, x_layout, output_layout, mode, theta,
        pass_ndims, rotary_emb_dim, static_cast<int32_t>(b), static_cast<int32_t>(m),
        static_cast<int32_t>(h), static_cast<int32_t>(k), static_cast<int32_t>(b_stride),
        static_cast<int32_t>(m_stride), static_cast<int32_t>(h_stride),
        static_cast<int32_t>(offset), static_cast<int32_t>(num_elements));
  } else {
    DispatchPackSize<T, PositionType, int64_t, num_dims>(
        stream, x, cos, sin, position_ids, out, position_shape, x_layout, output_layout, mode, theta,
        pass_ndims, rotary_emb_dim, b, m, h, k, b_stride, m_stride, h_stride, offset, num_elements);
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
    const int64_t pass_ndims = k_size - ctx->Attr<int64_t>("rotary_size");
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

    // TODO: hard code num_dims & seems redundant template problem...
    DispatchIndex<T, PositionType, ndims>(
        ctx->stream()->As<ep::CudaStream>(),
        reinterpret_cast<const T*>(x->dptr()),
        cos ? reinterpret_cast<const T*>(cos->dptr()) : nullptr,
        sin ? reinterpret_cast<const T*>(sin->dptr()) : nullptr,
        position_ids ? reinterpret_cast<const PositionType*>(position_ids->dptr()) : nullptr,
        reinterpret_cast<T*>(out->mut_dptr()),
        position_ids ? position_ids->shape_view().data() : nullptr, x_layout, output_layout, mode,
        static_cast<T>(theta), pass_ndims, rotary_emb_dim, b, m, h, k, b_stride, m_stride, h_stride,
        offset);
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
