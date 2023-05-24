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

#ifndef ONEFLOW_USER_KERNELS_CUFFT_PLAN_CACHE_H_
#define ONEFLOW_USER_KERNELS_CUFFT_PLAN_CACHE_H_

#include <cufft.h>
#include <cufftXt.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <functional>
#include <numeric>
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

constexpr int max_rank = 3;

enum class CUFFT_EXCUTETYPE { R2C, C2C, C2R };

struct CuFFTDataTypeDesc {
  cudaDataType inputtype;
  cudaDataType outputtype;
  cudaDataType executiontype;
};

}  // namespace

class CuFFTHandle {
  cufftHandle handle;

 public:
  CuFFTHandle() { OF_CUFFT_CHECK(cufftCreate(&handle)); }

  cufftHandle& get() { return handle; }
  const cufftHandle& get() const { return handle; }

  ~CuFFTHandle() { cufftDestroy(handle); }
};

// NOTE: The implementation of `CuFFTDataLayout`, `cufft_simple_embed` and `as_cufft_embed` are
// mostly taken from pytorch. For more details pls refer to `CuFFTPlanCache.h` in PyTorch.
typedef long long cufft_size_type;
typedef small_vector<cufft_size_type, max_rank + 1> cufft_dim_vector;
struct CuFFTDataLayout {
  small_vector<cufft_size_type, 5> embed;
  cufft_size_type stride, dist;
  bool must_clone, simple;
};

// Returns a cufft embedding for a contiguous signal of the given size.
// e.g. if the input is cloned, this will be the resulting data layout
inline CuFFTDataLayout cufft_simple_embed(const cufft_dim_vector& sizes, bool onesided) {
  CuFFTDataLayout layout;
  layout.simple = true;
  layout.must_clone = false;
  layout.embed.assign(sizes.cbegin() + 1, sizes.cend());
  if (onesided) { layout.embed.back() = sizes.back() / 2 + 1; }
  layout.stride = 1;
  layout.dist = 1;
  for (const auto& len : layout.embed) { layout.dist *= len; }
  return layout;
}

// Convert strides to a CuFFT embedded representation.
// If strides cannot be embedded, returns a simple layout and sets must_clone flag
inline CuFFTDataLayout as_cufft_embed(const cufft_dim_vector& strides,
                                      const cufft_dim_vector& sizes, bool onesided) {
  const auto signal_ndim = strides.size() - 1;
  CuFFTDataLayout layout;
  auto last_stride = strides[signal_ndim];
  layout.must_clone = (last_stride <= 0);

  const auto last_dim_size = onesided ? sizes[signal_ndim] / 2 + 1 : sizes[signal_ndim];

  const auto signal_numel = std::accumulate(sizes.begin() + 1, sizes.end() - 1, (cufft_size_type)1,
                                            std::multiplies<cufft_size_type>())
                            * last_dim_size;

  // Zero stides are not allowed, even if the batch size is one.
  // If that happens just set a dummy case
  if (sizes[0] == 1) {
    layout.dist = signal_numel;
  } else if (strides[0] == 0) {
    layout.must_clone = true;
  } else {
    layout.dist = strides[0];
  }

  // Calculate the embedding shape, or set must_clone if the strides cannot be embedded
  layout.embed.resize(signal_ndim);
  for (auto i = signal_ndim - 1; !layout.must_clone && i > 0; i--) {
    auto stride = strides[i];
    if (sizes[i] == 1) {
      layout.embed[i] = 1;
    } else if (stride > 0 && stride % last_stride == 0) {
      layout.embed[i] = stride / last_stride;
      last_stride = stride;
    } else {
      layout.must_clone = true;
    }
  }
  // must_clone == false
  if (layout.must_clone) {
    // If the input needs to be cloned, assume it will be contiguous
    layout = cufft_simple_embed(sizes, onesided);
    layout.must_clone = true;
  } else {
    layout.embed[0] = sizes[1];
    layout.stride = strides[signal_ndim];

    // Determine if layout represents a simple embedding (contiguous data)
    layout.simple = [&] {
      FOR_RANGE(int, i, 1, signal_ndim - 1) {
        if (layout.embed[i] != sizes[i + 1]) { return false; }
      }
      return (layout.stride == 1 && layout.dist == signal_numel
              && layout.embed.back() == last_dim_size);
    }();
  }
  return layout;
}

struct CuFFTParams {
  int64_t ndim;
  cufft_dim_vector input_shape;
  cufft_dim_vector input_strides;
  cufft_dim_vector output_shape;
  cufft_dim_vector output_strides;
  cufft_dim_vector data_shape;
  CUFFT_EXCUTETYPE excute_type;
  DataType real_data_type;

  CuFFTParams() = default;
  CuFFTParams(const Shape& in_shape, const Shape& out_shape, const Stride& in_strides,
              const Stride& out_strides, int64_t dims, CUFFT_EXCUTETYPE type, DataType real)
      : ndim(dims), excute_type(type), real_data_type(real) {
    CHECK_OR_THROW(ndim >= 1 && ndim <= max_rank);
    CHECK_OR_THROW(in_shape.size() == ndim + 1);
    CHECK_OR_THROW(out_shape.size() == ndim + 1);
    CHECK_OR_THROW(in_shape.size() == in_strides.size());
    CHECK_OR_THROW(out_shape.size() == out_strides.size());
    data_shape.resize(ndim + 1);
    input_shape.resize(in_shape.size());
    input_strides.resize(in_strides.size());
    output_shape.resize(out_shape.size());
    output_strides.resize(out_strides.size());

    std::copy(in_strides.begin(), in_strides.end(), input_strides.begin());
    std::copy(out_strides.begin(), out_strides.end(), output_strides.begin());
    std::copy(in_shape.begin(), in_shape.end(), input_shape.begin());
    std::copy(out_shape.begin(), out_shape.end(), output_shape.begin());

    data_shape[0] = input_shape[0];  // batch size
    FOR_RANGE(int64_t, i, 0, ndim) {
      auto in_size = input_shape[i + 1];
      auto out_size = output_shape[i + 1];
      data_shape[i + 1] = std::max(in_size, out_size);
      CHECK_OR_THROW(in_size == data_shape[i + 1] || in_size == (data_shape[i + 1] / 2) + 1);
      CHECK_OR_THROW(out_size == data_shape[i + 1] || out_size == (data_shape[i + 1] / 2) + 1);
    }
  }
};

class CuFFTConfig {
 public:
  CuFFTConfig(const CuFFTConfig&) = delete;
  CuFFTConfig& operator=(CuFFTConfig const&) = delete;
  ~CuFFTConfig() = default;

  explicit CuFFTConfig(CuFFTParams& params) {  // NOLINT

    if (params.real_data_type == kBFloat16 || params.real_data_type == kFloat16) {
      // CuFFT support half data type, but there are some limits:
      //  https://docs.nvidia.com/cuda/cufft/#half-precision-cufft-transforms
      CHECK_OR_THROW(false) << "Unsupported datatype kBFloat16 and kFloat16.";
    }

    CuFFTDataLayout input_layout = as_cufft_embed(params.input_strides, params.data_shape,
                                                  params.excute_type == CUFFT_EXCUTETYPE::C2R);
    CuFFTDataLayout output_layout = as_cufft_embed(params.output_strides, params.data_shape,
                                                   params.excute_type == CUFFT_EXCUTETYPE::R2C);

    bool clone_input = input_layout.must_clone;  // that means: input should be contiguous because
                                                 // original input can't be embeded
    const bool is_layout_simple = input_layout.simple && output_layout.simple;

    // disable cuFFT the default behavior of allocating work area at plan generating time
    OF_CUFFT_CHECK(cufftSetAutoAllocation(plan_handle_.get(), 0));
    infer_cufft_type_(params.excute_type, params.real_data_type);

    // exclude input_shape[0] whtich is batch dim
    cufft_dim_vector fft_shape(params.data_shape.begin() + 1, params.data_shape.end());
    cufft_size_type batch = params.data_shape[0];
    if (is_layout_simple) {
      OF_CUFFT_CHECK(cufftXtMakePlanMany(plan_handle_.get(), params.ndim, fft_shape.data(),
                                         /*inembed=*/nullptr, /*istride=*/1, /*idist=*/1,
                                         /*inputtype=*/data_type_desc_.inputtype,
                                         /*onembed=*/nullptr, /*ostride=*/1, /*odist=*/1,
                                         /*outputtype=*/data_type_desc_.outputtype,
                                         /*batch=*/batch, /*workSize=*/&work_size_,
                                         /*executiontype=*/data_type_desc_.executiontype));
    } else {
      OF_CUFFT_CHECK(cufftXtMakePlanMany(
          plan_handle_.get(), params.ndim, fft_shape.data(),
          /*inembed=*/input_layout.embed.data(), /*istride=*/input_layout.stride,
          /*idist=*/input_layout.dist, /*inputtype=*/data_type_desc_.inputtype,
          /*onembed=*/output_layout.embed.data(), /*ostride=*/output_layout.stride,
          /*odist=*/output_layout.dist, /*outputtype=*/data_type_desc_.outputtype,
          /*batch=*/batch, /*workSize=*/&work_size_,
          /*executiontype=*/data_type_desc_.executiontype));
    }
  }

  size_t workspace_size() const { return work_size_; }
  const cufftHandle& plan() const { return plan_handle_.get(); }

  void excute(void* input, void* output, bool forward) {
    OF_CUFFT_CHECK(
        cufftXtExec(plan_handle_.get(), input, output, forward ? CUFFT_FORWARD : CUFFT_INVERSE));
  }

 private:
  void infer_cufft_type_(CUFFT_EXCUTETYPE excute_type, DataType real_data_type) {
    if (real_data_type == kFloat) {
      data_type_desc_.executiontype = CUDA_C_32F;
      data_type_desc_.inputtype = excute_type == CUFFT_EXCUTETYPE::R2C ? CUDA_R_32F : CUDA_C_32F;
      data_type_desc_.outputtype = excute_type == CUFFT_EXCUTETYPE::C2R ? CUDA_R_32F : CUDA_C_32F;
    } else if (real_data_type == kDouble) {
      data_type_desc_.executiontype = CUDA_C_64F;
      data_type_desc_.inputtype = excute_type == CUFFT_EXCUTETYPE::R2C ? CUDA_R_64F : CUDA_C_64F;
      data_type_desc_.outputtype = excute_type == CUFFT_EXCUTETYPE::C2R ? CUDA_R_64F : CUDA_C_64F;
    } else {
      CHECK_OR_THROW(false) << "cuFFT doesn't support type " << real_data_type;
    }
  }

  CuFFTHandle plan_handle_;
  CuFFTDataTypeDesc data_type_desc_;
  size_t work_size_;
};

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_CUFFT_PLAN_CACHE_H_
