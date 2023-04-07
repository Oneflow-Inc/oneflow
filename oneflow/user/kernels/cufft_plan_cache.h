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
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

constexpr int max_rank = 3;

enum class CUFFT_EXCUTETYPE{ R2C, C2C, C2R };

struct CuFFT_DType_Desc{
  cudaDataType inputtype;
  cudaDataType outputtype;
  cudaDataType executiontype;
};

}


// NOTE: The implementation of `_cudaGetErrorEnum`  are mostly taken from
// pytorch.
//       For more details pls refer to:
//       https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/CuFFTUtils.h#L17
static inline std::string _cudaGetErrorEnum(cufftResult error)
{
  switch (error)
  {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
      return "CUFFT_INCOMPLETE_PARAMETER_LIST";
    case CUFFT_INVALID_DEVICE:
      return "CUFFT_INVALID_DEVICE";
    case CUFFT_PARSE_ERROR:
      return "CUFFT_PARSE_ERROR";
    case CUFFT_NO_WORKSPACE:
      return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED:
      return "CUFFT_NOT_IMPLEMENTED";
    case CUFFT_NOT_SUPPORTED:
      return "CUFFT_NOT_SUPPORTED";
    default:
      std::ostringstream ss;
      ss << "unknown error " << error;
      return ss.str();
  }
}

static inline void CUFFT_CHECK(cufftResult error)
{
  CHECK_OR_THROW(error == CUFFT_SUCCESS) << "cuFFT error: " << _cudaGetErrorEnum(error);
}

class CuFFTHandle{
  cufftHandle handle;
public:
  CuFFTHandle(){
    CUFFT_CHECK(cufftCreate(&handle));
  }

  cufftHandle* get(){
    return &handle;
  }
  const cufftHandle* get() const{
    return &handle;
  }

  ~CuFFTHandle(){
    cufftDestroy(handle);
  }
};

// NOTE: The implementation of `CuFFTDataLayout`, `cufft_simple_embed` and `as_cufft_embed` are mostly taken from
// pytorch.
//       For more details pls refer to:
//       https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/CuFFTPlanCache.h#L136
//       https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/CuFFTPlanCache.h#L145
//       https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/CuFFTPlanCache.h#L164
using cufft_size_type = long long int;
struct CuFFTDataLayout{
  small_vector<cufft_size_type, 5> embed;
  cufft_size_type stride, dist;
  bool must_clone, simple;
};

// Returns a cufft embedding for a contiguous signal of the given size.
// e.g. if the input is cloned, this will be the resulting data layout
inline CuFFTDataLayout cufft_simple_embed(const std::vector<cufft_size_type>& sizes, bool onesided) {
  CuFFTDataLayout layout;
  layout.simple = true;
  layout.must_clone = false;
  layout.embed.assign(sizes.cbegin() + 1, sizes.cend());
  if (onesided) {
    layout.embed.back() = sizes.back() / 2 + 1;
  }
  layout.stride = 1;
  layout.dist = 1;
  for (const auto& len : layout.embed) {
    layout.dist *= len;
  }
  return layout;
}

// Convert strides to a CuFFT embedded representation.
// If strides cannot be embedded, returns a simple layout and sets must_clone flag
inline CuFFTDataLayout as_cufft_embed(const std::vector<cufft_size_type>& strides, const std::vector<cufft_size_type>& sizes, bool onesided) {

  const auto signal_ndim = strides.size() - 1;
  CuFFTDataLayout layout;
  auto last_stride = strides[signal_ndim];
  layout.must_clone = (last_stride <= 0);

  const auto last_dim_size = onesided ?
      sizes[signal_ndim] / 2 + 1 : sizes[signal_ndim];
  // const auto signal_numel = c10::multiply_integers(sizes.slice(1, sizes.size() - 2)) * last_dim_size;
  const auto signal_numel = std::accumulate(sizes.begin() + 1, sizes.end() - 1, (cufft_size_type) 1, std::multiplies<cufft_size_type>()) * last_dim_size;
  // Zero stides are not allowed, even if the batch size is one.
  // If that happens just set a dummy case
  if (sizes[0] == 1) {
    layout.dist = signal_numel;
  } else if (strides[0] == 0) {
    layout.must_clone = true;
  } else {
    layout.dist = strides[0]; // 350
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
    layout.embed[0] = sizes[1]; // 10
    layout.stride = strides[signal_ndim]; // 1
    // Determine if layout represents a simple embedding (contiguous data)
    layout.simple = [&] {
      FOR_RANGE(int, i, 1, signal_ndim - 1){
        if (layout.embed[i] != sizes[i + 1]) {
          return false;
        }
      }
      // for (const auto i : c10::irange(1, signal_ndim - 1)) {
      //   if (layout.embed[i] != sizes[i + 1]) {
      //     return false;
      //   }
      // }
      return (layout.stride == 1 && layout.dist == signal_numel &&
          layout.embed.back() == last_dim_size);
    }();
  }
  return layout;
}

struct CuFFtParams {
  int32_t ndim;
  int32_t output_shape[max_rank + 1];
  int32_t input_shape[max_rank + 1];
  int32_t input_strides[max_rank + 1];
  int32_t output_strides[max_rank + 1];
  bool IsForward;
  CUFFT_EXCUTETYPE excute_type;
  DataType real_data_type;

  // int32_t* rank;
  // int32_t batch = 0;

  CuFFtParams() = default;
  CuFFtParams(const Shape& in_shape, const Shape& out_shape, const Stride& in_strides,
              const Stride& out_strides, int32_t dims, const bool is_forward,
              CUFFT_EXCUTETYPE type, DataType real) : ndim(dims), IsForward(is_forward), excute_type(type), real_data_type(real)
              {
        assert(ndim >= 1 && ndim <= max_rank);
        assert(in_shape.size() == in_strides.size());
        assert(out_shape.size() == out_strides.size());

        std::copy(in_strides.begin(), in_strides.end(), input_strides);
        std::copy(out_strides.begin(), out_strides.end(), output_strides);
        std::copy(in_shape.begin(), in_shape.end(), input_shape);
        std::copy(out_shape.begin(), out_shape.end(), output_shape);
  }
};

template<typename IN, typename OUT>
class CuFFtConfig {
 public:
  CuFFtConfig(const CuFFtConfig&) = delete;
  CuFFtConfig& operator=(CuFFtConfig const&) = delete;
  ~CuFFtConfig() = default;

  explicit CuFFtConfig(CuFFtParams& params) {  // NOLINT
    // cufftPlanMany(&plan_handle_, params.ndim, params.rank, params.input_shape,
    //               params.input_strides[0], params.input_strides[1], params.output_shape,
    //               params.output_strides[0], params.output_strides[1], exectype_, params.batch);

    if (params.real_data_type == kBFloat16 || params.real_data_type == kFloat16){
      // CuFFT support half data type, but there are some limits:
      //  https://docs.nvidia.com/cuda/cufft/#half-precision-cufft-transforms
      // TO-DO : do some check
    }
    

    infer_cufft_type_(params.excute_type, params.real_data_type);

    cufftXtMakePlanMany(&plan_handle_, params.ndim, params.input_shape, params.input_shape, 
          params.input_strides[0], long long idist, cudaDataType inputtype, 
          long long *onembed, long long ostride, long long odist, 
          cudaDataType outputtype, long long batch, size_t *workSize, 
          cudaDataType executiontype)
  }


 private:
  void infer_cufft_type_(CUFFT_EXCUTETYPE excute_type, DataType real_data_type) {
    if (real_data_type == kFloat16){
      data_type_desc.executiontype = CUDA_C_16F;
      data_type_desc.inputtype = excute_type == CUFFT_EXCUTETYPE::R2C ? CUDA_R_16F : CUDA_C_16F;
      data_type_desc.outputtype = excute_type == CUFFT_EXCUTETYPE::C2R ? CUDA_R_16F : CUDA_C_16F;
    }
    else if (real_data_type == kBFloat16){
      data_type_desc.executiontype = CUDA_C_16BF;
      data_type_desc.inputtype = excute_type == CUFFT_EXCUTETYPE::R2C ? CUDA_R_16BF : CUDA_C_16BF;
      data_type_desc.outputtype = excute_type == CUFFT_EXCUTETYPE::C2R ? CUDA_R_16BF : CUDA_C_16BF;
    }
    else if (real_data_type == kFloat){
      data_type_desc.executiontype = CUDA_C_32F;
      data_type_desc.inputtype = excute_type == CUFFT_EXCUTETYPE::R2C ? CUDA_R_32F : CUDA_C_32F;
      data_type_desc.outputtype = excute_type == CUFFT_EXCUTETYPE::C2R ? CUDA_R_32F : CUDA_C_32F;
    }
    else if (real_data_type == kDouble){
      data_type_desc.executiontype = CUDA_C_64F;
      data_type_desc.inputtype = excute_type == CUFFT_EXCUTETYPE::R2C ? CUDA_R_64F : CUDA_C_64F;
      data_type_desc.outputtype = excute_type == CUFFT_EXCUTETYPE::C2R ? CUDA_R_64F : CUDA_C_64F;
    }
    else{
      CHECK_OR_THROW(false) << "cuFFT doesn't support type " << real_data_type;
    }
  }

  CuFFTHandle plan_handle_;
  // cufftType cufft_exectype_;
  CuFFT_DType_Desc data_type_desc;
};

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_CUFFT_PLAN_CACHE_H_
